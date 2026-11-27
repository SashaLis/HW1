# Path строит пути относительно файла приложения, поэтому запуск возможен из другой папки.
from pathlib import Path
# pickle восстанавливает сохранённые в тетрадке модели и преобразования.
import pickle

# Altair описывает графики через колонки таблиц, оси, цвет и подсказки.
import altair as alt
# NumPy понадобится для границ интервалов и подсчёта частот.
import numpy as np
# pandas читает CSV и собирает таблицы для моделей и графиков.
import pandas as pd
# Под именем st доступны поля ввода, формы и вывод результатов Streamlit.
import streamlit as st

# Общие функции обработки используются и в приложении, и при проверке экспорта.
from inference import parse_measurements, predict_price

# __file__ указывает на app.py. parent даёт папку проекта с pickle и данными.
ROOT = Path(__file__).resolve().parent
# Основной цвет столбцов и линии обучения.
INK = '#284659'
# Второй цвет выделяет тестовую выборку, медианы или отрицательные веса.
COPPER = '#a6532b'
# Слева имена из модели, справа подписи для графика коэффициентов.
LABELS = {'year': 'Год выпуска', 'km_driven': 'Пробег', 'mileage': 'Экономичность',
          'engine': 'Объём двигателя', 'max_power': 'Мощность', 'name': 'Оценка марки'}
# Для каждого поля слева хранится исходное значение из CSV, справа русский текст для формы.
OPTIONS = {
    'fuel': {'Diesel': 'Дизель', 'Petrol': 'Бензин', 'CNG': 'Метан', 'LPG': 'Пропан-бутан'},
    'transmission': {'Manual': 'Механическая', 'Automatic': 'Автоматическая'},
    'seller_type': {'Individual': 'Частное лицо', 'Dealer': 'Дилер', 'Trustmark Dealer': 'Сертифицированный дилер'},
    'owner': {'First Owner': 'Первый', 'Second Owner': 'Второй', 'Third Owner': 'Третий',
              'Fourth & Above Owner': 'Четвёртый и последующие', 'Test Drive Car': 'Машина для тест-драйва'},
}


# Streamlit запускает скрипт при изменении виджета. Кеш сохраняет загруженные модели между запусками.
@st.cache_resource
def load_model(model_path, modified_ns):
    # modified_ns входит в ключ кеша. Если заменить pickle, Streamlit загрузит новые веса.
    with model_path.open('rb') as source:
        # Возвращаемые ключи совпадают с ключами bundle в последней ячейке тетрадки.
        bundle = pickle.load(source)
    # Метрики в pickle необязательны. Если их нет, один раз проверяем модели на размеченном тесте.
    if 'scores' not in bundle:
        test = pd.read_csv(ROOT / 'data/cars_test.csv')  # В CSV есть и характеристики, и реальные цены.
        actual = test.pop('selling_price')  # Отделяем ответы до вызова predict.
        bundle['scores'] = {}  # Каждая Ridge получит собственные показатели.
        for variant in bundle['models']:
            prediction = predict_price(bundle, test, variant)  # Один прогноз для каждой тестовой машины.
            errors = (actual - prediction).abs()  # Абсолютное отклонение в рупиях.
            bundle['scores'][variant] = {
                'mae': float(errors.mean()),  # Средняя абсолютная ошибка на всей выборке.
                'hits': int((errors <= 0.1 * actual).sum()),  # Количество попаданий в допуск 10%.
                'count': len(actual),  # Знаменатель для доли попаданий.
            }
    return bundle  # В кеше хранятся модели, предобработка и метрики.


# Кешируем таблицы, чтобы переключение между графиками не читало оба CSV заново.
@st.cache_data
def eda_samples():
    # В словаре будут две таблицы. По названию выборки интерфейс найдёт нужную.
    samples = {}
    # Для графиков обучения читаем исходные строки с ценами.
    train = pd.read_csv(ROOT / 'data/cars_train.csv')
    # Повтор определяем по всем характеристикам кроме цены, как в тетрадке. Сохраняется первая запись группы.
    train = train.drop_duplicates(subset=train.columns.drop('selling_price'))
    # Для теста берём все объявления согласно исходному разбиению.
    test = pd.read_csv(ROOT / 'data/cars_test.csv')
    # Обеим выборкам нужна одна обработка единиц измерения.
    for label, raw in [('Обучение', train), ('Тест', test)]:
        # После разбора чисел мощность и объём можно сравнивать и по ним можно считать корреляции.
        frame = parse_measurements(raw)
        # Подписываем таблицу для переключателя выборок и легенды.
        samples[label] = frame
    # Возвращаем обе подготовленные таблицы всем разделам EDA.
    return samples


# Одна функция создаёт поля для топлива, владельца, коробки и продавца.
def select(label, key):
    # key указывает, какой из справочников OPTIONS нужен этому полю.
    choices = OPTIONS[key]
    # Пользователь видит перевод, а форма получает исходный ключ для OneHotEncoder.
    return st.selectbox(label, list(choices), format_func=choices.get)


# Здесь из значений полей получается одна строка признаков и одна оценка цены.
def estimate(bundle, variant):
    # Форма отправляет значения всех полей вместе при нажатии кнопки.
    with st.form('car'):
        # Размещаем характеристики в двух колонках, чтобы форма поместилась на экране.
        left, right = st.columns(2)
        # В левой колонке собираем марку, год, пробег и характеристики двигателя.
        with left:
            # Оставляем поле name в общей схеме строки даже у модели, которая обходится без марки.
            name = ''
            # Балл марки влияет только на gsr, поэтому у gsr_wo это поле скрыто.
            if variant == 'gsr':
                # Список известных марок берём из pickle. Последний вариант даёт ввести машину другой марки.
                brands = sorted(bundle["prestige"]) + ['Другая марка']
                # index задаёт начальный выбор. В переменную попадает марка, которую выбрал пользователь.
                name = st.selectbox('Марка', brands, index=brands.index('Maruti'))
            # Границы соответствуют годам в обучении. Начальное значение формы равно 2014.
            year = st.number_input('Год выпуска', 1983, 2020, 2014)
            # Пробег передаём в километрах. Кнопки поля меняют число на тысячу.
            km = st.number_input('Пробег, км', min_value=0, value=60000, step=1000)
            # Выбранное топливо понадобится кодировщику категорий.
            fuel = select('Топливо', 'fuel')
            # Единица пробега на единицу топлива зависит от вида топлива.
            unit = 'км/л для бензина и дизеля, км/кг для газа'
            # Число из этого поля соответствует столбцу mileage исходного датасета.
            mileage = st.number_input(f'Экономичность, {unit}', min_value=0.0, value=20.0)
            # Для engine вводим объём в кубических сантиметрах.
            engine = st.number_input('Объём двигателя, см³', min_value=1, value=1248)
        # Справа размещаем оставшиеся характеристики машины и объявления.
        with right:
            # Модель обучалась на мощности в bhp, поэтому в форме та же единица.
            power = st.number_input('Мощность, bhp', min_value=0.1, value=74.0)
            # Выбор коробки передаётся модели под именем transmission.
            transmission = select('Коробка передач', 'transmission')
            # Варианты соответствуют числу мест в обучающей выборке. По умолчанию выбрано пять.
            seats = st.selectbox('Число мест', [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
            # Тип продавца отличает частное объявление от продажи через дилера.
            seller = select('Продавец', 'seller_type')
            # Эта категория описывает число владельцев или использование машины для тест-драйва.
            owner = select('Владелец по счёту', 'owner')
        # Возвращает True при отправке формы.
        submitted = st.form_submit_button('Рассчитать цену', type='primary')
    # Прогноз нужен только после того, как пользователь закончил ввод.
    if submitted:
        # Названия ключей повторяют колонки исходного CSV. Значения взяты из виджетов выше.
        row = dict(name=name, year=year, km_driven=km, fuel=fuel, mileage=mileage,
                   engine=engine, max_power=power, transmission=transmission,
                   seats=seats, seller_type=seller, owner=owner)
        # Квадратные скобки вокруг row создают таблицу из одной строки. Из массива ответов берём первый.
        value = predict_price(bundle, pd.DataFrame([row]), variant)[0]
        # У линейной регрессии нет нижней границы. Отдельно обрабатываем невозможную цену.
        if value > 0:
            # Для вывода округляем цену до рупии и разделяем тысячи пробелами.
            st.metric('Оценка цены', f'₹ {value:,.0f}'.replace(',', ' '))
        else:
            # Объясняем, почему в форме не показана цена.
            st.warning('Расчёт дал нулевую или отрицательную цену. Проверьте характеристики автомобиля.')
        # Для неизвестной марки стоит пояснить, какой балл повлиял на результат.
        if name == 'Другая марка':
            # Показываем фактическое запасное значение из этого файла модели.
            st.caption(f'Для этой марки модель использует медиану по обучению ({bundle["default_brand"]:g} балла).')
        # Метрики относятся к выбранной Ridge и ко всей тестовой выборке.
        score = bundle["scores"][variant]
        # Форматируем среднюю ошибку в рупиях для подписи.
        mae = format_number(score['mae'])
        # Сопоставляем прогноз конкретной машины с качеством модели на известных ценах.
        st.caption(f"Средняя абсолютная ошибка на тесте составила {mae} ₹. "
                   f"У {score['hits']} машин из {score['count']} ошибка не превысила 10%.")


# Два графика отвечают на вопросы о распределении цен и связи цены с мощностью.
def show_overview(data):
    # Этот заголовок относится к гистограмме ниже.
    st.subheader('Распределение цен')
    # len считает объявления после удаления повторов. Для тысяч используем пробел.
    count = f'{len(data):,}'.replace(',', ' ')
    # Объясняем размер выборки и способ чтения оси цены.
    st.caption(f'В обучающей выборке {count} объявлений после удаления повторов. '
               'Шкала цен логарифмическая. Одинаковый шаг означает рост цены в одно и то же число раз.')
    # Для гистограммы достаточно одного столбца фактических цен.
    prices = data.selling_price
    # Делим диапазон положительных цен на 31 интервал с равным шагом по логарифмической шкале.
    edges = np.geomspace(prices.min(), prices.max(), 32)
    # Считаем, сколько цен попало в каждый интервал. Границы уже есть в edges.
    counts, _ = np.histogram(prices, edges)
    # Каждая строка задаёт начало, конец и высоту одного столбца.
    bins = pd.DataFrame({'От': edges[:-1], 'До': edges[1:], 'Автомобилей': counts})
    # mark_bar выбирает столбцы, encode связывает их положение и размеры с данными.
    hist = alt.Chart(bins).mark_bar(color=INK).encode(
        # Тип Q означает число. bin='binned' сообщает, что границы уже вычислены.
        x=alt.X('От:Q', bin='binned', scale=alt.Scale(type='log'), title='Цена, ₹'),
        # x2 задаёт правый край столбца, y задаёт число автомобилей.
        x2='До:Q', y='Автомобилей:Q',
        # При наведении показываем обе границы и число попавших в них машин.
        tooltip=[alt.Tooltip('От', format=',.0f'), alt.Tooltip('До', format=',.0f'), 'Автомобилей'])
    # Рисуем готовую диаграмму по ширине раздела.
    st.altair_chart(hist, use_container_width=True)
    # Второй график сравнивает цены в группах мощности.
    st.subheader('Мощность и цена')
    # Берём пары «мощность и цена», где оба значения известны.
    points = pd.DataFrame({'Мощность': data.max_power, 'Цена': prices}).dropna()
    # Убираем нулевую мощность перед разбиением по интервалам.
    points = points[points['Мощность'] > 0]
    # pd.cut назначает каждой машине группу. Последняя включает всё, что выше 200 bhp.
    bucket = pd.cut(points['Мощность'], bins=[0, 50, 75, 100, 125, 150, 200, np.inf],
                    labels=['до 50', '50-75', '75-100', '100-125', '125-150', '150-200', 'свыше 200'])
    # Для непустых групп считаем медиану цены и число строк. reset_index возвращает группу в столбец.
    grouped = points.groupby(bucket, observed=True)['Цена'].agg(['median', 'size']).reset_index()
    # Для каждой группы нужен один столбец.
    chart = alt.Chart(grouped).mark_bar(color=INK).encode(
        # Тип N означает категорию. Сохраняем естественный порядок интервалов мощности.
        x=alt.X('Мощность:N', sort=grouped['Мощность'].tolist(), title='Мощность, bhp'),
        # Высота показывает медиану цены внутри группы.
        y=alt.Y('median:Q', title='Медианная цена, ₹'),
        # При наведении можно проверить цену и размер группы.
        tooltip=['Мощность', alt.Tooltip('median', title='Медиана, ₹', format=',.0f'),
                 alt.Tooltip('size', title='Автомобилей')])
    # Добавляем сравнение по мощности под гистограммой.
    st.altair_chart(chart, use_container_width=True)
    # Уточняем, какие объявления вошли в расчёт.
    st.caption('Наведите курсор на столбец, чтобы увидеть число машин. На график попали объявления с мощностью больше нуля.')


# Для анализа выбираем измеримые характеристики. Значения словаря содержат подписи и единицы.
EDA_LABELS = {
    'selling_price': 'Цена, ₹', 'year': 'Год выпуска', 'km_driven': 'Пробег, км',
    'max_power': 'Мощность, bhp', 'engine': 'Объём, см³', 'seats': 'Число мест',
}


# Сравниваем доли автомобилей в одинаковых группах обучения и теста.
def show_split_comparison(samples):
    # В выпадающем списке видны русские подписи, а key хранит имя столбца CSV.
    key = st.selectbox('Какой признак сравнить?', list(EDA_LABELS), format_func=EDA_LABELS.get)
    # Фиксированные границы имеют смысл в единицах признака и одинаковы для обеих выборок.
    thresholds = {
        'selling_price': [200_000, 400_000, 600_000, 1_000_000, 2_000_000],
        'year': [2000, 2005, 2010, 2015, 2018],
        'km_driven': [25_000, 50_000, 75_000, 100_000, 150_000],
        'max_power': [50, 75, 100, 125, 150, 200],
        'engine': [1000, 1250, 1500, 2000, 2500],
    }
    # У числа мест мало отдельных значений. Каждому даём свой ряд, в том числе редким.
    if key == 'seats':
        all_seats = pd.concat([data[key] for data in samples.values()]).dropna()
        groups = sorted(all_seats.astype(int).unique())
    else:
        edges = thresholds[key]
        # Подписываем крайние группы открытыми, чтобы включить самые большие и маленькие значения.
        labels = [str(v) if key == 'year' else f'{v:,}'.replace(',', ' ') for v in edges]
        groups = ([f'До {labels[0]}']
                  + [f'От {left} до {right}' for left, right in zip(labels, labels[1:])]
                  + [f'От {labels[-1]}'])

    rows = []
    # У каждой выборки свои медиана и знаменатель при переводе количества в проценты.
    for column, (label, data) in zip(st.columns(2), samples.items()):
        values = data[key].dropna()
        median = f'{values.median():,.1f}'.removesuffix('.0').replace(',', ' ').replace('.', ',')
        column.metric(f'Медиана ({label.lower()})', median)
        column.caption(f'Значение указано в {len(values)} из {len(data)} объявлений')
        if key == 'seats':
            buckets = values.astype(int)
        else:
            # right=False включает нижнюю границу. Ровно 400 000 попадает в группу от 400 000.
            buckets = pd.cut(values, [-np.inf, *edges, np.inf], labels=groups, right=False)
        # Пустая группа получает ноль, так что у обучения и теста совпадают все строки диаграммы.
        counts = buckets.value_counts().reindex(groups, fill_value=0)
        rows.append(pd.DataFrame({'Группа': counts.index.astype(str), 'Выборка': label,
                                  'Автомобилей': counts.to_numpy(), 'Доля': counts.to_numpy() / len(values)}))

    # Длина полосы показывает долю, две полосы одной группы располагаются рядом по вертикали.
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


# Здесь сравниваем разброс цен внутри категорий одной выбранной выборки.
def show_group_prices(data):
    # Пользователь выбирает один из четырёх категориальных признаков.
    groups = {'transmission': 'Коробка передач', 'fuel': 'Топливо',
              'seller_type': 'Продавец', 'owner': 'Владелец по счёту'}
    # Получаем имя поля, по которому нужно разделить объявления.
    key = st.selectbox('Группировать по признаку', list(groups), format_func=groups.get)
    # Собираем цены отдельно для каждого значения выбранного поля.
    by_group = data.groupby(key)['selling_price']
    # Считаем три квартиля. unstack раскладывает их по столбцам одной строки на группу.
    summary = by_group.quantile([.25, .5, .75]).unstack()
    # Даём полученным столбцам имена для графика и подсказок.
    summary.columns = ['Q1', 'Медиана', 'Q3']
    # Количество цен помогает оценить, насколько представлена каждая категория.
    summary['Автомобилей'] = by_group.count()
    # Переводим категории с помощью OPTIONS и превращаем индекс в столбец «Группа».
    summary = summary.rename(index=OPTIONS[key]).rename_axis('Группа').reset_index()
    # Сначала показываем группы с самой высокой медианной ценой.
    order = summary.sort_values('Медиана', ascending=False)['Группа'].tolist()
    # Оба слоя диаграммы используют одну таблицу и одни строки по вертикали.
    base = alt.Chart(summary).encode(
        # Для длинных названий групп резервируем место в подписях оси.
        y=alt.Y('Группа:N', sort=order, title=None, axis=alt.Axis(labelLimit=180)),
        # По наведению видны размер группы, границы интервала и медиана.
        tooltip=['Группа', 'Автомобилей', alt.Tooltip('Q1:Q', title='Первый квартиль, ₹', format=',.0f'),
                 alt.Tooltip('Медиана:Q', title='Медиана, ₹', format=',.0f'),
                 alt.Tooltip('Q3:Q', title='Третий квартиль, ₹', format=',.0f')])
    # Рисуем отрезок от первого до третьего квартиля.
    span = base.mark_rule(strokeWidth=5, color=INK).encode(x=alt.X('Q1:Q', title='Цена, ₹'), x2='Q3:Q')
    # На тот же ряд накладываем точку в положении медианы.
    median = base.mark_point(filled=True, size=110, fill=COPPER, stroke='white').encode(x='Медиана:Q')
    # Знак + накладывает точку на отрезок. Для каждой группы выделяем место по высоте.
    st.altair_chart((span + median).properties(height=max(160, len(summary) * 44)), use_container_width=True)
    # Подсказываем, как читать точку и границы отрезка.
    st.caption('Точка отмечает медианную цену. Отрезок охватывает половину цен группы, от первого до третьего квартиля. '
               'Наведите курсор, чтобы узнать, сколько в ней машин.')


# Строим матрицу парных корреляций выбранной выборки.
def show_correlations(data):
    # Оставляем числовые поля и сразу переводим их названия для осей.
    numeric = data[list(EDA_LABELS)].rename(columns=EDA_LABELS)
    # По умолчанию pandas считает корреляцию Пирсона по известным значениям каждой пары.
    corr = numeric.corr()
    # Одинаковый порядок на осях помещает корреляцию каждого признака с самим собой на диагональ.
    axis_order = list(numeric)
    # melt превращает матрицу в таблицу «строка, столбец, r». Одной записи соответствует одна клетка.
    pairs = corr.rename_axis('Строка').reset_index().melt(
        id_vars='Строка', var_name='Столбец', value_name='r')
    # Цветные клетки и текст должны располагаться на одинаковых осях.
    base = alt.Chart(pairs).encode(
        # Наклоняем нижние подписи, чтобы названия не перекрывались.
        x=alt.X('Столбец:N', sort=axis_order, title=None, axis=alt.Axis(labelAngle=-32)),
        # По вертикали располагаем второй признак пары.
        y=alt.Y('Строка:N', sort=axis_order, title=None),
        # При наведении значение можно прочитать с точностью до тысячных.
        tooltip=['Строка', 'Столбец', alt.Tooltip('r:Q', format='.3f', title='Корреляция')])
    # Каждую пару признаков изображаем прямоугольником.
    cells = base.mark_rect().encode(color=alt.Color(
        # Задаём разные цвета для отрицательной и положительной корреляции со светлым нулём между ними.
        'r:Q', scale=alt.Scale(domain=[-1, 0, 1], range=[COPPER, '#f3f3f1', INK]), legend=None))
    # Поверх цвета пишем число с двумя знаками после запятой.
    labels = base.mark_text().encode(text=alt.Text('r:Q', format='.2f'),
                                    # Контраст текста выбираем по силе корреляции, чтобы число читалось на тёмной клетке.
                                    color=alt.condition('abs(datum.r) > 0.65', alt.value('white'), alt.value('#142d38')))
    # Накладываем числа на клетки и выводим матрицу.
    st.altair_chart((cells + labels).properties(height=250), use_container_width=True)
    # Удаляем саму цену, иначе максимумом всегда будет 1. Ищем самую сильную связь по модулю.
    strongest = corr['Цена, ₹'].drop('Цена, ₹').abs().idxmax()
    # Берём исходный коэффициент, чтобы в выводе сохранить его знак.
    r = corr.loc[strongest, 'Цена, ₹']
    # Для русской подписи меняем десятичную точку на запятую.
    coefficient = f'{r:.3f}'.replace('.', ',')
    # Сообщаем результат именно для той выборки, которую сейчас смотрит пользователь.
    st.caption(f'У признака «{strongest}» самая высокая по модулю корреляция с ценой (r = {coefficient}). '
               'Для каждой пары признаков в расчёт вошли только объявления, где указаны оба значения.')


# Переключатель раздела данных определяет, какие таблицы попадут в график.
def show_data():
    # Получаем обе выборки из кеша Streamlit.
    samples = eda_samples()
    # Выбор вопроса определяет, какой график нужно построить.
    view = st.selectbox('Что исследовать?', ['Обзор цен и мощности', 'Обучение и тест',
                                            'Цены по категориям', 'Корреляции признаков'])
    # Обзор в тетрадке делался на обучении после удаления повторов.
    if view == 'Обзор цен и мощности':
        # Передаём в обзор одну таблицу.
        show_overview(samples['Обучение'])
    # Для прямого сравнения одновременно нужны обучение и тест.
    elif view == 'Обучение и тест':
        # Передаём словарь с обеими таблицами.
        show_split_comparison(samples)
    else:
        # Для групп и корреляций пользователь сам выбирает одну выборку.
        sample = st.radio('Выборка', list(samples), horizontal=True)
        # Категориальный признак требует группировки цен.
        if view == 'Цены по категориям':
            # Передаём только выбранные объявления.
            show_group_prices(samples[sample])
        else:
            # В последнем режиме строим корреляции на тех же выбранных строках.
            show_correlations(samples[sample])


# Общий формат чисел для коэффициентов и оценки ошибки.
def format_number(value, signed=False):
    # Флаг signed включает плюс у положительных чисел. Обе маски оставляют два десятичных знака.
    spec = '+,.2f' if signed else ',.2f'
    # Заменяем английские разделители тысяч и дробной части на привычные русские.
    return format(value, spec).replace(',', ' ').replace('.', ',')


# Показываем веса той же Ridge, которую пользователь выбрал для прогноза.
def show_weights(bundle, variant):
    regression = bundle['models'][variant]
    # Индекс связывает каждый вес с именем столбца, использованного при обучении.
    weights = pd.Series(regression.coef_, index=regression.feature_names_in_)
    st.write(f'Свободный член: **{format_number(regression.intercept_)} ₹**')

    # Привязываем подписи к именам после кодирования, как при обучении модели.
    labels = dict(LABELS)
    encoder = bundle['encoder']
    for field, categories in zip(encoder.feature_names_in_, encoder.categories_):
        for raw in categories:
            if field == 'seats':
                label = f'Мест в салоне: {raw}'
            else:
                label = OPTIONS.get(field, {}).get(raw, str(raw))
            labels[f'{field}_{raw}'] = label

    # Одинаковый масштаб в обеих колонках позволяет сравнивать длину полос.
    limit = weights.abs().max() or 1
    scale = alt.Scale(domain=[-limit, limit], nice=False)
    ownership = weights.index.str.startswith(('owner_', 'seats_'))
    groups = [('Автомобиль и продажа', weights[~ownership]), ('Владелец и места', weights[ownership])]

    for column, (title, group) in zip(st.columns(2), groups):
        column.markdown(f'**{title}**')
        table = group.rename_axis('Поле').reset_index(name='Вес')
        table['Признак'] = table['Поле'].replace(labels)
        table['Вес, ₹'] = table['Вес'].map(lambda value: format_number(value, signed=True))
        # Порядок модели сохраняет рядом категории топлива, владельца и числа мест.
        rows = alt.Chart(table).encode(
            y=alt.Y('Поле:N', sort=table['Поле'].tolist(), axis=None),
            tooltip=['Признак:N', 'Вес, ₹:N'],
        )
        # Название и число стоят над полосой, поэтому даже короткие полосы легко прочитать.
        bars = rows.mark_bar(size=3, yOffset=8).encode(
            x=alt.X('Вес:Q', scale=scale, axis=None),
            color=alt.condition(alt.datum.Вес < 0, alt.value(COPPER), alt.value(INK)),
        )
        names = rows.mark_text(align='left', dy=-3, fontSize=12).encode(
            x=alt.value(0), text='Признак:N',
        )
        values = rows.mark_text(align='right', dy=-3, fontSize=12).encode(
            x=alt.value('width'), text='Вес, ₹:N',
        )
        chart = (bars + names + values).properties(height=alt.Step(26)).configure_view(stroke=None)
        column.altair_chart(chart, use_container_width=True)

    st.caption('Веса указаны в рупиях. Для числовых признаков шаг равен одному стандартному отклонению. '
               'Для категорий вес показывает разницу с базовым вариантом.')


# Отсюда начинается выполнение интерфейса. Streamlit повторяет этот блок при взаимодействии с виджетами.
st.set_page_config(page_title='Цена автомобиля', layout='centered')
# Общий заголовок виден на любом экране.
st.title('Цена подержанного автомобиля')
# Главное меню возвращает название раздела, выбранного пользователем.
section = st.radio('Раздел', ['Оценить автомобиль', 'Данные', 'Коэффициенты'],
                   horizontal=True, label_visibility='collapsed')
# Разделу с графиками нужны наблюдения из CSV.
if section == 'Данные':
    # Передаём управление выбору графика.
    show_data()
else:
    # Внутренние ключи совпадают с ключами models в pickle. Первый вариант выбран по умолчанию.
    variants = {'gsr_wo': 'Без рейтинга марки', 'gsr': 'С рейтингом марки'}
    # Загружаем словарь для прогноза или коэффициентов.
    model_path = ROOT / 'model.pickle'
    # Время изменения файла отличает новую выгрузку от уже закешированной.
    bundle = load_model(model_path, model_path.stat().st_mtime_ns)
    # В переменной variant остаётся ключ модели, а в интерфейсе видно её название.
    variant = st.selectbox('Модель Ridge', list(variants), format_func=variants.get, key='variant')
    # По выбранному разделу определяем, нужна ли форма.
    if section == 'Оценить автомобиль':
        # Уточняем валюту и диапазон лет в обучающих данных.
        st.caption('Цены в индийских рупиях. В обучении есть машины с 1983 по 2020 год выпуска.')
        # Показываем поля ввода и обрабатываем отправку формы.
        estimate(bundle, variant)
    else:
        # На другом экране выводим веса той же модели.
        show_weights(bundle, variant)
