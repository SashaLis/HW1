from pathlib import Path
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def strange_cases(x):
    try:
        return float(x.split()[0])
    except:
        return np.NaN

def parse_measurements(frame):
    result = frame.copy()
    for column in ['mileage', 'engine', 'max_power']:
        result.loc[~result[column].isna(), column] = (
            result.loc[~result[column].isna(), column].apply(lambda x: strange_cases(x)))
        result[column] = result[column].astype('float')
    return result


def predict_price(bundle, raw, variant):
    data = raw.fillna(bundle['medians'])
    data[['seats', 'engine']] = data[['seats', 'engine']].astype(int)
    data['name'] = data['name'].map(bundle['prestige']).fillna(bundle['default_brand'])

    scaler = bundle['scaler']
    encoder = bundle['encoder']
    numeric_columns = scaler.feature_names_in_
    category_columns = encoder.feature_names_in_
    scaled = pd.DataFrame(scaler.transform(data[numeric_columns]), columns=numeric_columns)
    encoded = pd.DataFrame(encoder.transform(data[category_columns]),
                           columns=encoder.get_feature_names_out())
    features = pd.concat([scaled, encoded], axis=1)
    model = bundle['models'][variant]
    return model.predict(features[model.feature_names_in_])

# Общие функции обработки используются и в приложении, и при проверке экспорта.
ROOT = Path(__file__).resolve().parent

INK = '#284659'
COPPER = '#a6532b'

LABELS = {'year': 'Год выпуска', 'km_driven': 'Пробег', 'mileage': 'Экономичность',
          'engine': 'Объём двигателя', 'max_power': 'Мощность', 'name': 'Оценка марки'}

OPTIONS = {
    'fuel': {'Diesel': 'Дизель', 'Petrol': 'Бензин', 'CNG': 'Метан', 'LPG': 'Пропан-бутан'},
    'transmission': {'Manual': 'Механическая', 'Automatic': 'Автоматическая'},
    'seller_type': {'Individual': 'Частное лицо', 'Dealer': 'Дилер', 'Trustmark Dealer': 'Сертифицированный дилер'},
    'owner': {'First Owner': 'Первый', 'Second Owner': 'Второй', 'Third Owner': 'Третий',
              'Fourth & Above Owner': 'Четвёртый и последующие', 'Test Drive Car': 'Машина для тест-драйва'},
}


# Кеш сохраняет загруженные модели между запуск
@st.cache_resource
def load_model(model_path):
    with model_path.open('rb') as source:
        bundle = pickle.load(source)
    return bundle


# Кешируем таблицы, чтобы переключение между графиками не читало оба CSV заново.
@st.cache_data
def eda_samples():
    samples = {}
    train = pd.read_csv(ROOT / 'data/cars_train.csv')
    train = train.drop_duplicates(subset=train.columns.drop('selling_price'), keep = 'first')
    test = pd.read_csv(ROOT / 'data/cars_test.csv')
    for label, raw in [('Обучение', train), ('Тест', test)]:
        frame = parse_measurements(raw)
        samples[label] = frame
    return samples

def select(label, key):
    choices = OPTIONS[key]
    return st.selectbox(label, list(choices), format_func=choices.get)


def estimate(bundle, variant):
    with st.form('car'):
        left, right = st.columns(2)
        with left:
            name = ''
            if variant == 'gsr':
                brands = sorted(bundle["prestige"]) + ['Другая марка']
                name = st.selectbox('Марка', brands, index=brands.index('Hyundai'))

            year = st.number_input('Год выпуска', 1983, 2020, 2014)
            km = st.number_input('Пробег, км', min_value=0, value=60000, step=1000)
            fuel = select('Топливо', 'fuel')
            mileage = st.number_input('Экономичность', min_value=0.0, value=20.0)
            engine = st.number_input('Объём двигателя, см³', min_value=1, value=1248)

        with right:
            power = st.number_input('Мощность, bhp', min_value=0.1, value=74.0)
            transmission = select('Коробка передач', 'transmission')
            seats = st.selectbox('Число мест', [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
            seller = select('Продавец', 'seller_type')
            owner = select('Владелец по счёту', 'owner')
        submitted = st.form_submit_button('Рассчитать цену', type='primary')

    if submitted:
        row = dict(name=name, year=year, km_driven=km, fuel=fuel, mileage=mileage,
                   engine=engine, max_power=power, transmission=transmission,
                   seats=seats, seller_type=seller, owner=owner)
        value = predict_price(bundle, pd.DataFrame([row]), variant)[0]
        if value > 0:
            st.metric('Оценка цены', f'{value:,.0f}'.replace(',', ' '))
        else:
            st.warning('Расчёт дал нулевую или отрицательную цену. Проверьте характеристики автомобиля.')
        if name == 'Другая марка':
            st.caption(f'Для этой марки модель использует медиану по обучению ({bundle["default_brand"]:g} балла).')


def show_overview(data):
    st.subheader('Распределение цен')
    count = f'{len(data):,}'.replace(',', ' ')
    st.caption(f'В обучающей выборке {count} объявлений после удаления повторов')
    prices = data.selling_price
    edges = np.geomspace(prices.min(), prices.max(), 32)
    counts, _ = np.histogram(prices, edges)
    bins = pd.DataFrame({'От': edges[:-1], 'До': edges[1:], 'Автомобилей': counts})
    hist = alt.Chart(bins).mark_bar(color=INK).encode(
        x=alt.X('От:Q', bin='binned', scale=alt.Scale(type='log'), title='Цена'),
        x2='До:Q', y='Автомобилей:Q',
        tooltip=[alt.Tooltip('От', format=',.0f'), alt.Tooltip('До', format=',.0f'), 'Автомобилей'])
    st.altair_chart(hist, use_container_width=True)


    st.subheader('Мощность и цена')
    points = pd.DataFrame({'Мощность': data.max_power, 'Цена': prices}).dropna()
    points = points[points['Мощность'] > 0]
    bucket = pd.cut(points['Мощность'], bins=[0, 50, 75, 100, 125, 150, 200, np.inf],
                    labels=['до 50', '50-75', '75-100', '100-125', '125-150', '150-200', 'свыше 200'])
    grouped = points.groupby(bucket, observed=True)['Цена'].agg(['median', 'size']).reset_index()
    chart = alt.Chart(grouped).mark_bar(color=INK).encode(
        x=alt.X('Мощность:N', sort=grouped['Мощность'].tolist(), title='Мощность, bhp'),
        y=alt.Y('median:Q', title='Медианная цена'),
        tooltip=['Мощность', alt.Tooltip('median', title='Медиана', format=',.0f'),
                 alt.Tooltip('size', title='Автомобилей')])
    st.altair_chart(chart, use_container_width=True)
    st.caption('Наведите курсор на столбец, чтобы увидеть число машин. На график попали объявления с мощностью больше нуля.')


EDA_LABELS = {
    'selling_price': 'Цена', 'year': 'Год выпуска', 'km_driven': 'Пробег, км',
    'max_power': 'Мощность, bhp', 'engine': 'Объём, см³', 'seats': 'Число мест',
}


def show_split_comparison(samples):
    key = st.selectbox('Какой признак сравнить?', list(EDA_LABELS), format_func=EDA_LABELS.get)
    thresholds = {
        'selling_price': [200_000, 400_000, 600_000, 1_000_000, 2_000_000],
        'year': [2000, 2005, 2010, 2015, 2018],
        'km_driven': [25_000, 50_000, 75_000, 100_000, 150_000],
        'max_power': [50, 75, 100, 125, 150, 200],
        'engine': [1000, 1250, 1500, 2000, 2500],
    }

    if key == 'seats':
        all_seats = pd.concat([data[key] for data in samples.values()]).dropna()
        groups = sorted(all_seats.astype(int).unique())
    else:
        edges = thresholds[key]
        labels = [str(v) if key == 'year' else f'{v:,}'.replace(',', ' ') for v in edges]
        groups = ([f'До {labels[0]}']
                  + [f'От {left} до {right}' for left, right in zip(labels, labels[1:])]
                  + [f'От {labels[-1]}'])

    rows = []

    for column, (label, data) in zip(st.columns(2), samples.items()):
        values = data[key].dropna()
        median = f'{values.median():,.1f}'.removesuffix('.0').replace(',', ' ').replace('.', ',')
        column.metric(f'Медиана ({label.lower()})', median)
        column.caption(f'Значение указано в {len(values)} из {len(data)} объявлений')
        if key == 'seats':
            buckets = values.astype(int)
        else:
            buckets = pd.cut(values, [-np.inf, *edges, np.inf], labels=groups, right=False)
        counts = buckets.value_counts().reindex(groups, fill_value=0)
        rows.append(pd.DataFrame({'Группа': counts.index.astype(str), 'Выборка': label,
                                  'Автомобилей': counts.to_numpy(), 'Доля': counts.to_numpy() / len(values)}))

    chart = alt.Chart(pd.concat(rows, ignore_index=True)).mark_bar().encode(
        y=alt.Y('Группа:N', sort=[str(v) for v in groups], title=EDA_LABELS[key],
                axis=alt.Axis(labelLimit=180)),
        yOffset=alt.YOffset('Выборка:N', sort=list(samples)),
        x=alt.X('Доля:Q', title='Доля автомобилей', axis=alt.Axis(format='.0%')),
        color=alt.Color('Выборка:N', scale=alt.Scale(domain=list(samples), range=[INK, COPPER]),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['Группа:N', 'Выборка:N', 'Автомобилей:Q', alt.Tooltip('Доля:Q', format='.1%')],
    ).properties(height=alt.Step(18))
    st.altair_chart(chart, use_container_width=True)
    st.caption('Проценты рассчитаны отдельно для каждой выборки по объявлениям с известным значением. '
               'Количество машин видно при наведении.')
    if key != 'seats':
        st.caption('Верхняя граница каждого интервала входит в следующую группу.')



def show_group_prices(data):
    groups = {'transmission': 'Коробка передач', 'fuel': 'Топливо',
              'seller_type': 'Продавец', 'owner': 'Владелец по счёту'}
    key = st.selectbox('Группировать по признаку', list(groups), format_func=groups.get)
    by_group = data.groupby(key)['selling_price']
    summary = by_group.quantile([.25, .5, .75]).unstack()
    summary.columns = ['Q1', 'Медиана', 'Q3']
    summary['Автомобилей'] = by_group.count()
    summary = summary.rename(index=OPTIONS[key]).rename_axis('Группа').reset_index()
    order = summary.sort_values('Медиана', ascending=False)['Группа'].tolist()
    base = alt.Chart(summary).encode(
        y=alt.Y('Группа:N', sort=order, title=None, axis=alt.Axis(labelLimit=180)),
        tooltip=['Группа', 'Автомобилей', alt.Tooltip('Q1:Q', title='Первый квартиль', format=',.0f'),
                 alt.Tooltip('Медиана:Q', title='Медиана', format=',.0f'),
                 alt.Tooltip('Q3:Q', title='Третий квартиль', format=',.0f')])
    span = base.mark_rule(strokeWidth=5, color=INK).encode(x=alt.X('Q1:Q', title='Цена'), x2='Q3:Q')
    median = base.mark_point(filled=True, size=110, fill=COPPER, stroke='white').encode(x='Медиана:Q')
    st.altair_chart((span + median).properties(height=max(160, len(summary) * 44)), use_container_width=True)
    st.caption('Точка отмечает медианную цену. Отрезок охватывает половину цен группы, от первого до третьего квартиля. '
               'Наведите курсор, чтобы узнать, сколько в ней машин.')


def show_correlations(data):
    numeric = data[list(EDA_LABELS)].rename(columns=EDA_LABELS)
    corr = numeric.corr()
    axis_order = list(numeric)
    pairs = corr.rename_axis('Строка').reset_index().melt(
        id_vars='Строка', var_name='Столбец', value_name='r')
    base = alt.Chart(pairs).encode(
        x=alt.X('Столбец:N', sort=axis_order, title=None, axis=alt.Axis(labelAngle=-32)),
        y=alt.Y('Строка:N', sort=axis_order, title=None),
        tooltip=['Строка', 'Столбец', alt.Tooltip('r:Q', format='.3f', title='Корреляция')])
    cells = base.mark_rect().encode(color=alt.Color(
        'r:Q', scale=alt.Scale(domain=[-1, 0, 1], range=[COPPER, '#f3f3f1', INK]), legend=None))
    labels = base.mark_text().encode(text=alt.Text('r:Q', format='.2f'),
                                    color=alt.condition('abs(datum.r) > 0.65', alt.value('white'), alt.value('#142d38')))
    st.altair_chart((cells + labels).properties(height=250), use_container_width=True)
    strongest = corr['Цена'].drop('Цена').abs().idxmax()
    r = corr.loc[strongest, 'Цена']
    coefficient = f'{r:.3f}'.replace('.', ',')
    st.caption(f'У признака «{strongest}» самая высокая по модулю корреляция с ценой (r = {coefficient}). '
               'Для каждой пары признаков в расчёт вошли только объявления, где указаны оба значения.')


def show_data():
    samples = eda_samples()
    view = st.selectbox('Что исследовать?', ['Обзор цен и мощности', 'Обучение и тест',
                                            'Цены по категориям', 'Корреляции признаков'])
    if view == 'Обзор цен и мощности':
        show_overview(samples['Обучение'])
    elif view == 'Обучение и тест':
        show_split_comparison(samples)
    else:
        sample = st.radio('Выборка', list(samples), horizontal=True)
        if view == 'Цены по категориям':
            show_group_prices(samples[sample])
        else:
            show_correlations(samples[sample])



def format_number(value, signed=False):
    spec = '+,.2f' if signed else ',.2f'
    return format(value, spec).replace(',', ' ').replace('.', ',')


def show_weights(bundle, variant):
    regression = bundle['models'][variant]
    weights = pd.Series(regression.coef_, index=regression.feature_names_in_)
    st.write(f'Свободный член: **{format_number(regression.intercept_)}**')

    labels = dict(LABELS)
    encoder = bundle['encoder']
    for field, categories in zip(encoder.feature_names_in_, encoder.categories_):
        for raw in categories:
            if field == 'seats':
                label = f'Мест в салоне: {raw}'
            else:
                label = OPTIONS.get(field, {}).get(raw, str(raw))
            labels[f'{field}_{raw}'] = label

    limit = weights.abs().max() or 1
    scale = alt.Scale(domain=[-limit, limit], nice=False)
    ownership = weights.index.str.startswith(('owner_', 'seats_'))
    groups = [('Автомобиль и продажа', weights[~ownership]), ('Владелец и места', weights[ownership])]

    for column, (title, group) in zip(st.columns(2), groups):
        column.markdown(f'**{title}**')
        table = group.rename_axis('Поле').reset_index(name='Вес')
        table['Признак'] = table['Поле'].replace(labels)
        table['Вес'] = table['Вес'].map(lambda value: format_number(value, signed=True))
        rows = alt.Chart(table).encode(
            y=alt.Y('Поле:N', sort=table['Поле'].tolist(), axis=None),
            tooltip=['Признак:N', 'Вес:N'],
        )
        bars = rows.mark_bar(size=3, yOffset=8).encode(
            x=alt.X('Вес:Q', scale=scale, axis=None),
            color=alt.condition(alt.datum.Вес < 0, alt.value(COPPER), alt.value(INK)),
        )
        names = rows.mark_text(align='left', dy=-3, fontSize=12).encode(
            x=alt.value(0), text='Признак:N',
        )
        values = rows.mark_text(align='right', dy=-3, fontSize=12).encode(
            x=alt.value('width'), text='Вес:N',
        )
        chart = (bars + names + values).properties(height=alt.Step(26)).configure_view(stroke=None)
        column.altair_chart(chart, use_container_width=True)



st.set_page_config(page_title='Цена автомобиля', layout='centered')
st.title('Цена автомобиля')
section = st.radio('Раздел', ['Оценить автомобиль', 'Данные', 'Коэффициенты'],
                   horizontal=True, label_visibility='collapsed')

if section == 'Данные':
    show_data()
else:
    variants = {'gsr_wo': 'Без рейтинга марки', 'gsr': 'С рейтингом марки'}
    model_path = ROOT / 'model.pickle'
    bundle = load_model(model_path)
    variant = st.selectbox('Модель Ridge', list(variants), format_func=variants.get, key='variant')
    if section == 'Оценить автомобиль':
        st.caption('В обучении есть машины с 1983 по 2020 год выпуска.')
        estimate(bundle, variant)
    else:
        show_weights(bundle, variant)
