# -*- coding: utf-8 -*-
"""AI_HW1_Regression_with_inference_base.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XbFOvw98xAn78qRsZhds4JvEKNLhxOWu

<a href="https://colab.research.google.com/github/Murcha1990/ML_AI24/blob/main/Hometasks/Base/AI_HW1_Regression_with_inference_base.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Домашнее задание №1 (base)**

В этом домашнем задании вам будет необходимо:
*  обучить модель регрессии для предсказания стоимости автомобилей;
* реализовать веб-сервис для применения построенной модели на новых данных

**Максимальная оценка за дз**
> Оценка за домашку = $min(\text{ваш балл}, 11)$

**Мягкий дедлайн: 27 ноября 23:59**

**Жесткий дедлайн: 20 декабря 23:59 (конец модуля)**

**Примечание**

В каждой части оцениваются как код, **так и ответы на вопросы.** Вопросы подсвечены синим цветом.

Если нет одного и/или другого, то часть баллов за соответствующее задание снимается.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

random.seed(42)
np.random.seed(42)

"""**Задание 0 (0 баллов).**
Для чего фиксируем сиды в домашках?

`Your answer here`

# **Часть 1 | EDA и визуализация**

Первая часть состоит из классических шагов EDA:

- Базовый EDA и обработка признаков (2.5 балла)
- Визуализации признаков и их анализ (1 балл)

Всего можно набрать 3.5 основных балла и 0.65 бонусных. Бонусные задания выделены как **Дополнительное задание/Бонус**. Вы можете выполнять их, чтобы в случае ошибок в основных задачах всё равно набрать за работу максимум. Кроме того, дополнительные задания позволяют вам углубить знания.

Призываем активно использовать их!

## **Простейший EDA и обработка признаков (2.5 балла)**
"""

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

print("Train data shape:", df_train.shape)
print("Test data shape: ", df_test.shape)

"""### **Задание 1 (0.55 балла)**

Выполните операции, направленные на практику основных действий с `pandas`:
- [ ] Отобразите 10 случайных строк тренировочного датасета (0.15 балла)
- [ ] Отобразите первые 5 и последние 5 объектов тестового датасета (0.1 балла)
- [ ] Посмотрите, есть ли в датасете пропуски. Запишите/выведите названия колонок, для которых есть пропущенные значения (0.1 балла)
- [ ] Посмотрите, есть ли в данных явные дубликаты (0.05 балла)


**Бонус**
- [ ] Постройте дашборд, используя [ydata-profilling](https://github.com/ydataai/ydata-profiling)(0.15 балла)
"""

df_train.shape
number_rows = df_train.shape[0]
number_columns = df_train.shape[1]
print('Количечество строк:', number_rows)
print('Количество столбцов:', number_columns)

df_test.shape
number_rows = df_test.shape[0]
number_columns = df_test.shape[1]
print('Количечество строк:', number_rows)
print('Количество столбцов:', number_columns)

# Отобразите 10 случайных строк тренировочного датасета (0.15 балла)
df_train.sample(n=10, random_state=36)

#Отобразите первые 5  объектов тестового датасета (0.1 балла)
df_test.head()

#Отобразите последние 5 объектов тестового датасета (0.1 балла)
df_test.tail()

#Посмотрите, есть ли в датасете пропуски. Запишите/выведите названия колонок, для которых есть пропущенные значения (0.1 балла)
total = df_train.isnull().sum().sum()
print("Общее количество пропусков в датасете (df_train):", total)

column = df_train.isnull().sum()
m_columns = column[column > 0]
print(m_columns)

#Посмотрите, есть ли в датасете пропуски. Запишите/выведите названия колонок, для которых есть пропущенные значения (0.1 балла)
total = df_test.isnull().sum().sum()
print("Общее количество пропусков в датасете (df_test):", total)

column = df_test.isnull().sum()
m_columns = column[column > 0]
print(m_columns)

#Посмотрите, есть ли в данных явные дубликаты (0.05 балла)
df_train.duplicated().sum()
d = df_train.duplicated().sum()
print('Количество дублей в датасете (df_train):', d)

#Посмотрите, есть ли в данных явные дубликаты (0.05 балла)
df_test.duplicated().sum()
d = df_test.duplicated().sum()
print('Количество дублей в датасете (df_test):', d)

#Установим ydata-profiling
!pip install -U ydata-profiling

#Постройте дашборд, используя ydata-profilling(0.15 балла)
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
profile_report = ProfileReport(df_train, title='Pandas Profiling Report')

"""Мы обнаружили пропуски. Давайте избавимся от них.

- [ ] Заполните пропуски в столбцах медианами. Убедитесь, что после заполнения пропусков не осталось. Заполнение пропусков проводите для обоих наборов данных, если необходимо

**Важно!**

При заполнении пропусков и в тестовом, и тренировочном наборах данных вы определяетесь со стратегией предобработки пропущенных значений при потенциальной работе модели.

Так как в теоретическом случае вы не имеете доступа к тестовой выборке, то заполняемой значение (у нас — медиана) вы считаете про *тренировочному* набору данных и им же заполняете *тестовый*.
"""

# Необходимо выясниь типы данных для корерктного анализа данных.
columns_with_na = df_train.isna().sum()[df_train.isna().sum() > 0].index.to_list()
df_train[columns_with_na].dtypes

# Необходимо выясниь типы данных для корерктного анализа данных.
columns_with_na = df_test.isna().sum()[df_test.isna().sum() > 0].index.to_list()
df_test[columns_with_na].dtypes

import pandas as pd
import numpy as np
# находим целые числа и числа с плавающей запятой используя регулярные выражения.
df_train['mileage'] = df_train['mileage'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)
#удаляем подстроку ' CC' из строк в столбце engine. Здесь regex=False указывает, что мы не используем регулярные выражения.
#.astype(float) преобразует оставшиеся строки (числовые значения) в формат float.
df_train['engine'] = df_train['engine'].str.replace(' CC', '', regex=False).astype(float)
# аналогично предыдущему. удаляем " bhp"
df_train['max_power'] = pd.to_numeric(df_train['max_power'].str.replace(' bhp', '', regex=False), errors='coerce')
#тоже самое проделывем с df_test
df_test['mileage'] = df_test['mileage'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)
df_test['engine'] = df_test['engine'].str.replace(' CC', '', regex=False).astype(float)
df_test['max_power'] = pd.to_numeric(df_test['max_power'].str.replace(' bhp', '', regex=False), errors='coerce')

# Параметр torque имеет разные единицы измерения (например: 160Nm@ 2000rpm/22.4 kgm at 1750-2750rpm). Необходимо привести к единой системе измерения (Нм)
def parse_torque(value):
    if pd.notna(value):
        kgm_w = pd.Series(value).str.extract(r'(\d+(\.\d+)?)\s*kgm')
        nm_w = pd.Series(value).str.extract(r'(\d+(\.\d+)?)\s*Nm')
        if not kgm_w.empty and kgm_w[0].notna().any():
            kgm_value = float(kgm_w[0].iloc[0])
            return kgm_value * 9.81
        elif not nm_w.empty and nm_w[0].notna().any():
            return float(nm_w[0].iloc[0])
    return None

df_train['torque'] = df_train['torque'].apply(parse_torque).astype(float)
df_test['torque'] = df_test['torque'].apply(parse_torque).astype(float)

columns_with_na = df_train.isna().sum()[df_train.isna().sum() > 0].index.to_list()
df_train[columns_with_na].dtypes

df_test.head()

#При заполнении пропусков и в тестовом, и тренировочном наборах данных вы определяетесь со стратегией предобработки пропущенных значений при потенциальной работе модели.
#Так как в теоретическом случае вы не имеете доступа к тестовой выборке,
#то заполняемой значение (у нас — медиана) вы считаете про тренировочному набору данных и им же заполняете тестовый.

import warnings
warnings.filterwarnings("ignore")

# вместо пропусков подставим медианные значения
mileage_mean = df_train['mileage'].astype(float).median()
engine_mean = df_train['engine'].astype(float).median()
df_train['max_power'].replace('', np.nan, inplace=True)
max_power_mean = df_train['max_power'].astype(float).median()
torque_mean = df_train['torque'].astype(float).median()
seats_mean = df_train.seats.mean()
df_train.loc[df_train['mileage'].isna(), 'mileage'] = mileage_mean
df_train.loc[df_train['engine'].isna(), 'engine'] = engine_mean
df_train.loc[df_train['max_power'].isna(), 'max_power'] = max_power_mean
df_train.loc[df_train['torque'].isna(), 'torque'] = torque_mean
df_train.loc[df_train['seats'].isna(), 'seats'] = seats_mean
# подставим вместо пропусков медианным значением из df_train в df_test
df_test.loc[df_test['mileage'].isna(), 'mileage'] = mileage_mean
df_test.loc[df_test['engine'].isna(), 'engine'] = engine_mean
df_test.loc[df_test['max_power'].isna(), 'max_power'] = max_power_mean
df_test.loc[df_test['torque'].isna(), 'torque'] = torque_mean
df_test.loc[df_test['seats'].isna(), 'seats'] = seats_mean

#посмотрим остались ли пропуски
total = df_train.isnull().sum().sum()
print("Общее количество пропусков в датасете (df_train):", total)

column = df_train.isnull().sum()
m_columns = column[column > 0]
print(m_columns)

"""### **Задание 2 (0.5 балла)**

- [ ] Посмотрите, есть ли в трейне объекты с одинаковым признаковым описанием (целевую переменную следует исключить). Если есть, то сколько? (0.1 балла)
- [ ] Отобразите такие объекты (0.15 балла)
- [ ] Удалите повторяющиеся строки. Если при одинаковом признаковом описании цены на автомобили отличаются, то оставьте первую строку по этому автомобилю (0.15 балла)
- [ ]  Обновите индексы строк таким образом, чтобы они шли от 0 без пропусков (0.1 балла)

"""

#Посмотрите, есть ли в трейне объекты с одинаковым признаковым описанием (целевую переменную следует исключить). Если есть, то сколько? (0.1 балла)
duplicates = df_train[df_train.drop('selling_price',axis=1).duplicated()]
d= duplicates.shape[0]
print('Количество дублей в датасете (df_train):', d)

#Отобразите такие объекты (0.15 балла)
duplicates.head()

#Удалите повторяющиеся строки. Если при одинаковом признаковом описании цены на автомобили отличаются, то оставьте первую строку по этому автомобилю (0.15 балла)
df_train = df_train.drop_duplicates(subset=df_train.drop('selling_price',axis=1).columns)

assert df_train.shape == (5840, 13)

df_train.reset_index(drop=True, inplace=True)

"""Отлично! Мы избавились от маленьких и явных проблем. Теперь перейдем к более сложным недостаткам полученной таблицы.

### **Задание 3 (0.25 балла)**

Вы могли заметить, что с признаками ``mileage, engine, max_power и torque`` всё не очень хорошо. Они распознаются как строки (можно убедиться в этом, вызвав `data.dtypes`). Однако эти переменные не являются категориальными — они — числа. Соответственно, нужно привести их к числовому виду.

**Задача :**
* [ ] Уберите единицы измерения для признаков ``mileage, engine, max_power``.
* [ ] Приведите тип данных к ``float``.
* [ ] Удалите столбец ``torque``


**Важно**
- Все действия нужно производить над обоими датасетами — `train` и `test`.
- Стобец ``torque`` мы удаляем для простоты. В идеальном случае, его также стоило бы предобработать.
"""

#В задачи 1 прописан код для следущих задач:
#-Уберите единицы измерения для признаков mileage, engine, max_power.
#-Приведите тип данных к float.
#Стобец torque был предобработан

"""### **Задание 4 (0.1 балла)**

Теперь, когда не осталось пропусков, давайте преобразуем столбцы к более подходящим типам. А именно столбцы ``engnine`` и ``seats`` к приведем к `int`.

- [ ] Осуществите приведение столбцов к необходимому типу.
"""

# your code here
df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)
df_test['engine'] = df_test['engine'].astype(int)
df_test['seats'] = df_test['seats'].astype(int)

"""### **Задание 5 (0.1 балла)**

Отлично! Мы провели "косметическую" предобработку и теперь готовы сделать важный шаг в контексте анализа данных. А именно — посмотреть на статистики!

**Ваша задача:**
- [ ] Посчитайте основные статистики по числовым столбцам для трейна и теста
- [ ] Посчитайте основные статистики по категориальным столбцам для трейна и теста

**Подсказка:**

Используте ``.describe()`` с нужным(и) аргументом(-ами).

**Примечание:**

Более корректно рассматривать статистики до заполнения пропусков и после, чтобы убедиться, что мы не внесли каких-либо серьезных сдвигов в изначальные рапсределения.
"""

df_train.describe(include='number')

df_test.describe(include='number')

df_train.describe(include='object')

df_test.describe(include='object')

assert df_train.shape == (5840, 13)

"""## **Визуализации (1 балл + 0.5 бонус)**

Визуализация данных — важный шаг в работе. Визуализировать данные необходимо, например, чтобы:

- Оценить распределения признаков самих по себе (это может натоклнуть вас на мысли о модели, которую можно использовать)
- Сравнить распределения на `train` и `test` — чтобы проверить, насколько информация, на которой вы будете обучаться согласуется с той, на которой модель должна работать
- Оценить есть ли явная связь признаков с целевой переменной

**Важно:**

Если распределения на `train` и `test` не совпадают, это не значит, что нужно перемешивать данные! Более корректно актуализировать задачу и уточнить, а не устарели ли данные `train`. Также полезным может быть собрать новую тестовую выборку, смешав те, что имеются сейчас.

**Если вы будете подгонять распределения, то можете встретиться с переобучением!**

### **Задание 6 (0.5 балла)**

Шаг 1.
- [ ] Воспользуйтесь `pairplot` из библиотеки `seabron`, чтобы визуализировать попарные распределения числовых признаков для `train`
- [ ] По полученному графику ответьте на вопросы:
 - Можно ли предположить на основе распределений связь признаков с целевой переменной?
 - Можно ли предположить на основе распределений выдвинуть гипотезу о корреляциях признаков?

Шаг 2.

- [ ] Постройте pairplot по тестовым данным
- [ ] Ответьте на вопрос "Похожими ли оказались совокупности при разделении на трейн и тест?"
"""

sns.pairplot(df_train)
plt.show()

sns.pairplot(df_test)
plt.show()

"""Наблюдаем следующие связи целевой переменной (selling price) с факторными переменными:


1.   Прямая связь между ценой и годом выпуска автомобиля (новая машина стоит дороже)
2.   прямая связь цены с мощностью, объемом двигателя и крутящим моментом. Чем больше данные показатели, тем дороже автомобиль.
3. Обратная связь между ценой и количеством мест. На графике наблюдаем снижение цены при увеличении посадочных мест свыше 5. Возможно на это так же влияет низкая мощность и крутящий момент при объеме двигателя от 2000 до 3000куб.м.
Скорее всего автомобиль с большим количесвом посадочных мест относится к семейному классу так как наблюдается небольшой пробег. Если машина использоавлся в рабочих целях то пробег был бы сущесвтенно больше.
4. обратная связь между ценой и пробегом (чем выше пробег, тем дешевле машина).
5. Наблюда слабую кореляюцию между ценой и расходом топлива.

### **Задание 7 (0.5 балла)**

И так, вы выдвинули гипотезы о наличии связи. Теперь давайте оценим эту связь в числах.

**Задание:**
- [ ] Получите значения коэффициента корреляции Пирсона для тренировочного набора данных при помощи `pd.corr()`
- [ ] По полученным корреляциям постройте тепловую карту (`heatmap` из бибилотеки seaborn)
"""

corr_map = df_train.select_dtypes(include='number').corr()
sns.heatmap(corr_map, annot=True, fmt='.1f')
plt.show()

"""- [ ] Ответьте на вопросы:
 - Какие 2 признака наименее скоррелированы между собой?
 - Между какими наблюдается довольно сильная положительная линейная зависимость?
 - Правильно ли, опираясь на данные, утверждать, что чем меньше год, тем, скорее всего, больше километров проехала машина к дате продажи?

1.Наименее скоррелированы между собой два показателя это год и объем двигателя (коэффициент корреляции = 0).
Это связано с тем, что каждый год выпускаются почти одинаковые наборы двигателей по объему для удволетворения всех классов потребителей (и соответсственно не наблюдается что в какой-то год стали пользоваться поплуярностью двигатели с определнным объемом).

2. Сильно скоррелированы мощность, крутящий момент,объем двигателя, а так же мощност и цена. Все факторы очевидно взаимосвязаны.
Для построения модели можно использовать фактор - мощность, так как наиболее скоррелирован с ценой.

3. Да, можно утверждать что чем старше автомобиль, тем выше пробег. Об обратной связи свидетельствует отрицательный коэффициент корреляции = -0.37.

### **Бонус (0.5 балла)**

Если вам кажется, что мы не попросили вас нарисовать какие-то очень важные зависимости, нарисуйте их **и поясните.**
"""

plt.scatter(x=df_train['fuel'], y=df_train['max_power'])
plt.xlabel('fuel')
plt.ylabel('max_power')
plt.show()

"""На графике видно, что мощные двигатели в основном используют Дизельное топливо и бензин. Двигатели на газе не дают высокую мощность. Это связано с тем что при сгорании газа выделяется меньше теплоты.

# **Часть 2 | Модель только на вещественных признаках**

В этой части вам предстоит обучить модель только на вещественных признаках. Почему только на них?

Чем больше признаковое пространство — чем сложнее модель. А чем модель проще — тем лучше для скорости работы и интерпретации признаков.

За задания этой части вы можете набрать 1.25 балла;

### **Задание 8 (0.05 балла)**

Разбейте данные на тренировочный и тестовый наборы. Перед разбиением создайте копию датафрейма, который будет хранить только вещественные признаки и используйте его (то есть категориальные столбцы (все, кроме seats) необходимо удалить).

В переменные y_train и y_test запишите значения целевых переменных.
"""

y_train = df_train['selling_price']
X_train = df_train[['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']]

assert X_train.shape == (5840, 6)

y_test = df_test['selling_price']
X_test = df_test[['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']]

assert X_test.shape == (1000, 6)

"""### **Задание 9 (0.2 балла)**

Построим нашу первую модель!
- [ ] Обучите классическую линейную регрессию с дефолтными параметрами. Посчтитайте $R^2$ и $MSE$ для трейна и для теста.
- [ ] Сделайте выводы по значениям метрик качества.

**Примечание:**

Здесь и далее $R^2$ и $MSE$ для трейна и для теста выводите везде, где требуется обучать модели, даже если в явном виде этого не просят. Иначе непонятно, как понять, насколько успешны наши эксперименты.
"""

from logging import LogRecord
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_lr = lr.predict(X_train)
y_test_lr = lr.predict(X_test)
mse_train = MSE(y_train, y_train_lr)
rsq_train = r2_score(y_train, y_train_lr) * 100
mse_test = MSE(y_test, y_test_lr)
rsq_test = r2_score(y_test, y_test_lr) * 100
print(f'Result:')
print(f'Mean Squared Error (train): {mse_train:,.1f}')
print(f'Mean Squared Error (test):  {mse_test:,.1f}')
print(f'Rsquared (train):  {rsq_train:.1f}%')
print(f'Rsquared (test):   {rsq_test:.1f}%')

"""### **Задание 10 (0.15 балла)**

Всегда есть место совершенству. Поэтому давайте попробуем улучшить модель. При помощи стандартизации признаков.

- [ ] Стандартизируйте значения в тренировочных и тестовых данных. Стандартизатор **обучайте только на `train`**.
"""

from sklearn.preprocessing import StandardScaler
# your code here
scaler = StandardScaler()
scaler.fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)
lr.fit(X_train_trans, y_train)
y_train_trans = lr.predict(X_train_trans)
y_test_trans = lr.predict(X_test_trans)
mse_train_trans = MSE(y_train, y_train_trans)
rsq_train_trans = r2_score(y_train, y_train_trans) * 100
mse_test_trans = MSE(y_test, y_test_trans)
rsq_test_trans = r2_score(y_test, y_test_trans) * 100
print(f'Result:')
print(f'Mean Squared Error (train):   {mse_train_trans:,.1f}')
print(f'Mean Squared Error (test):    {mse_test_trans:,.1f}')
print(f'Rsquared (train):  {rsq_train_trans:.1f}%')
print(f'Rsquared (test):   {rsq_test_trans:.1f}%')

"""Изменение не наблюдается.

### **Задание 11 (0.1 балла)**

Хотя стандартизация не помогла сильно прибавить в качестве она открыла возможность интерпретировать важность признаков в модели. Правило интерпретации такое:

Чем больше коэффициент $\beta_i$ по модулю, тем важнее признак.

**Ответьте на вопрос:**

- [ ] Какой признак оказался наиболее информативным в предсказании цены?
"""

# your code here
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
scaler.fit(X_train)
X_train_trans = scaler.transform(X_train)
lr.fit(X_train_trans, y_train)
y_train_trans = lr.predict(X_train_trans)

lr = LinearRegression()
lr.fit(X_train_trans, y_train_trans)

coefficients = lr.coef_
print("Коэффициенты:", coefficients)

feature_names = X_train.columns

# Нахождение индекса максимального коэффициента
max_index = np.argmax(coefficients)

# Вывод признака с наибольшим коэффициентом
print(f'Наиболее информативным признаком в предсказании цены с наибольшим коэффициентом является: {feature_names[max_index]} (коэффициент: {coefficients[max_index]})')

"""### **Задание 12 (0.25 балла)**

Попробуем улучшить нашу модель с помощью применения регуляризации. Для этого воспльзуемся `Lasso` регрессией.  Кроме того, попробуйте использовать её теоретическое свойство отбора признаков, за счет зануления незначимых коэффициентов.

**Задание:**

- [ ] Обучите Lasso регрессию на тренировочном наборе данных с нормализованными признаками. Оцените её качество
- [ ] Проверьте, занулила ли L1-регуляризация с параметрами по умолчанию какие-нибудь веса? Предположите почему.
"""

from sklearn.linear_model import Lasso

# your code here
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=5)
lasso.fit(X_train_trans, y_train)
y_train_lasreg = lasso.predict(X_train_trans)
y_test_lasreg = lasso.predict(X_test_trans)
mse_train_lasreg = MSE(y_train, y_train_lasreg)
rsq_train_lasreg = r2_score(y_train, y_train_lasreg) * 100
mse_test_lasreg = MSE(y_test, y_test_lasreg)
rsq_test_lasreg = r2_score(y_test, y_test_lasreg) * 100
print(f'Lasso result:')
print(f'Mean Squared Error train: {mse_train_lasreg:,.1f}')
print(f'Mean Squared Error test:  {mse_test_lasreg:,.1f}')
print(f'Rsquared train:  {rsq_train_lasreg:.1f} %')
print(f'Rsquared test:   {rsq_test_lasreg:.1f} %')

# your code here

print(lasso.coef_)

"""Модель не занулила веса.
 Оставшиеся признаки существены для прогноза.
 Увеличение Альфа до 5 не меняет ситуацию.
"""



"""### **Задание 13 Финальный рывок (0.5 балла)**

До этого мы с вами использовали `train` для обучения и `test` для прогнозирования. Но у нас есть ещё одна задача — подобрать оптимальные параметры модели. Для этого используем кросс-валидацию, описанную на семинарах.

Кроме того, выжмем максимум из модификаций регрессии. Построим `ElasticNet`. И сделаем всё по порядку.

**Ваша задача 1:**

- [ ] Перебором по сетке (c 10-ю фолдами) подберите оптимальные параметры для Lasso-регрессии. Вам пригодится класс [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [ ] Ответьте на вопросы:
 - Сколько грид-сёрчу пришлось обучать моделей?
 - Какой коэффициент регуляризации у лучшей из перебранных моделей? Занулились ли какие-нибудь из весов при такой регуляризации?
"""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# your code here
lasso = Lasso()
pipeline = Pipeline([('scaler', scaler), ('lasso', lasso)])
param_grid = {
    'lasso__alpha': np.logspace(-2, 4, 400)
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f'Result:')
print(f"The best alpha: {grid_search.best_params_['lasso__alpha']:.1f}")
print(f"The best Mean Squared Error:   {-grid_search.best_score_:,.1f}")
print(f"Regression coefficients: {grid_search.best_estimator_.named_steps['lasso'].coef_}")

# your code here
l = grid_search.best_params_['lasso__alpha']
lasso = Lasso(alpha= l)
lasso.fit(X_train_trans, y_train)
y_train_lasreg = lasso.predict(X_train_trans)
y_test_lasreg = lasso.predict(X_test_trans)
mse_train_lasreg = MSE(y_train, y_train_lasreg)
rsq_train_lasreg = r2_score(y_train, y_train_lasreg) * 100
mse_test_lasreg = MSE(y_test, y_test_lasreg)
rsq_test_lasreg = r2_score(y_test, y_test_lasreg) * 100
print(f'Lasso')
print(f'Mean Squared Error  train: {mse_train_lasreg:,.1f}')
print(f'Mean Squared Error  test:  {mse_test_lasreg:,.1f}')
print(f'Rsquared train:  {rsq_train_lasreg:.1f}%')
print(f'Rsquared test:   {rsq_test_lasreg:.1f}%')

"""При значении 400 параметра alpha и cv=10 в диапазонее 10^-2 и 10^4 = 4 000 моделей.
Лучший коэфициент регуляризации = 2256
Значение MSE получилось хуже, чем при alpha = 5. КОэфициенты не занулились.





"""



import numpy as np
n_alpha_values = 400  # np.logspace(-2, 4, 400)
n_folds = 10  #  cv=10
total_models = n_alpha_values * n_folds
print(f"Общее количество моделей, которые будут обучены: {total_models}")

"""**Ваша задача 2:**

- [ ] Перебором по сетке (c 10-ю фолдами) подберите оптимальные параметры для [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) регрессии.
- [ ] Ответьте на вопрос:
 - Сколько грид-сёрчу пришлось обучать моделей?
 - Какие гиперпараметры соответствуют лучшей (по выбранной метрике качества) из перебранных моделей?
"""

from sklearn.linear_model import ElasticNet
# your code here
elc = ElasticNet()
pipeline = Pipeline([
    ('scaler', scaler),
    ('elc', elc)
])
param_grid = {
    'elc__alpha': np.logspace(-2, 4, 300),
    'elc__l1_ratio': np.linspace(0, 1, 10)
}
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10,
                           scoring='neg_mean_squared_error', n_jobs=-1,
                           verbose=0)
grid_search.fit(X_train, y_train)
print(f"The best parameter")
print(f"The best alpha:    {grid_search.best_params_['elc__alpha']:.1f}")
print(f"The best l1_ratio: {grid_search.best_params_['elc__l1_ratio']:.1f}")
print(f"The best Mean Squared Error:      {-grid_search.best_score_:,.1f}")
print(f"Regression coefficients: {grid_search.best_estimator_.named_steps['elc'].coef_}")

import numpy as np
n_alpha_values = 300  # np.logspace(-2, 4, 300)
n_folds = 10  #  cv=10
total_models = n_alpha_values * n_folds
print(f"Общее количество моделей, которые будут обучены: {total_models}")

"""1.   Пришлось обучать 3000 моделей np.logspace(-2, 4,300), сv=10.
2.   Лучший параметр l1_ratio: 1.00 - => значение совпадает с лассо-регуляризацией расчитанным ранее с параметром np.logspace(-2, 4, 300).

# **Часть 3| Добавляем категориальные фичи**

Попробуем для улучшения модели дать ей больше признаков. Добавим категориальные фичи.

За эту часть можно набрать 0.75 основных балла и 0.25 бонусных.

### **Задание 14 (0.1 балла)** Проанализируйте столбец `name`. Очевидно, что эта переменная является категориальной, однако категорий в ней много.

В этом домашнем задании мы предлагаем удалить его.

**Ваша задача:**
- [ ] Удалить столбец`name`
"""

# your code here
X_train_categor = df_train.drop(['name', 'selling_price'], axis=1)
X_test_categor = df_test.drop(['name', 'selling_price'], axis=1)

"""В другом случае, конечно, мы могли бы предобработать данный столбец. В качестве бонуса предлагаем вам придумать и реализовать алгоритм предобработки.

### **Бонус 0.5 балла**
- [ ] Предобработайте столбец `name`, чтобы избежать его удаления
"""

X_train_categor_name = df_train.drop('selling_price', axis=1).copy()

X_train_categor_name.name.value_counts()

"""можно раздеить на бренд автомобиля и модель бренда."""

X_train_categor_name[['car_brand', 'model']] = X_train_categor_name['name'].str.split(n=2, expand=True)[[0, 1]]
X_train_categor_name.drop(columns=['name'], inplace=True)



X_train_categor_name.car_brand.value_counts()[:10]

X_train_categor_name.model.value_counts()[:10]

columns_with_na = df_train.isna().sum()[df_train.isna().sum() > 0].index.to_list()
df_train[columns_with_na].dtypes

"""### **Задание 15 (0.4 балла)**

- [ ] Закодируйте категориалльные фичи и ``seats`` методом OneHot-кодирования. Обратите внимание, что во избежание мультиколлинеарности следует избавиться от одного из полученных столбцов при кодировании каждого признака методом OneHot.
"""

from sklearn.preprocessing import OneHotEncoder # или можно использовать get_dummies из библиотеки pandas

# your code here
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
columns_scaling = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque']
cat_t = OneHotEncoder(drop='first')
num_t = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_t, features),
        ('num', num_t, columns_scaling)
    ])
X_train_t = preprocessor.fit_transform(X_train_categor)
X_test_t = preprocessor.transform(X_test_categor)
X_train_t_data = pd.DataFrame(X_train_t)
X_test_t_data = pd.DataFrame(X_test_t)

"""### **Задание 16 (0.25 балла)**
Повторим то, что делали на прошлом шаге для моделей на вещественных признаках, однако теперь с моделью `Ridge`.


**Ваша задача:**
- [ ] Переберите параметр регуляризации `alpha` для гребневой (ridge) регрессии с помощью класса `GridSearchCV` В качестве параметров при объявлении GridSearchCV кроме модели укажите метрику качества $R^2$. Кроссвалидируйтесь по 10-ти фолдам.
- [ ] Ответье на вопрос: Удалось ли улучшить качество прогнозов?
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# your code here
ridgereg = Ridge()
param_grid = {
    'alpha': np.logspace(-3, 3, 100)
}
gs = GridSearchCV(estimator=ridgereg, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1)
gs.fit(X_train_t_data, y_train)
rsquared = gs.best_score_
ridge_alpha = gs.best_params_['alpha']
print(f"Показатели:")
print(f"The best alpha: {ridge_alpha:.5f}")
print(f"The best Rsquared: {rsquared:.2f}")

ridgereg = Ridge(alpha=gs.best_params_['alpha'])
ridgereg.fit(X_train_t_data, y_train)
y_train_ridgereg = ridgereg.predict(X_train_t_data)
y_test_ridgereg = ridgereg.predict(X_test_t_data)

mse_train_ridgereg = MSE(y_train, y_train_ridgereg)
rsq_train_ridgereg = r2_score(y_train, y_train_ridgereg) * 100
mse_test_ridgereg = MSE(y_test, y_test_ridgereg)
rsq_test_ridgereg = r2_score(y_test, y_test_ridgereg) * 100
print(f'Показатели')
print(f'Mean Squared Error  train: {mse_train_ridgereg:,.1f}')
print(f'Mean Squared Error  test:  {mse_test_ridgereg:,.1f}')
print(f'Rsquared train:  {rsq_train_ridgereg:.1f}%')
print(f'Rsquared test:   {rsq_test_ridgereg:.1f}%')

"""Качество модели лучше:
Rsquared test увеличился с 57% до 65%, ean Squared Error  test: снизился до 201 с 247 млн.
"""



"""# **Часть 4. | Бизнесовая (0.5 балла)**

### **Задание 17 (0.5 балла)**

В мире бизнеса очень важно давать оценку качества модели понятную бизнесу, поэтому иногда заказчики приходят с кастомными метриками. Попробуем сделать такую для нашей задачи.

**Описание метрики:**

Среди всех предсказанных цен на авто нужно посчитать долю прогнозов, отличающихся от реальных цен на эти авто не более чем на 10% (в одну или другую сторону)

**Ваша задача:**

- [ ] Реализуйте метрику `business_metric`
- [ ] Посчитайте метрику для всех обученных моделей и определеите, какаю лучше всего решает задачу бизнеса
"""

# your code here
import numpy as np
def business_metric(y_true, y_pred):
  assert y_true.shape[0] == y_pred.shape[0], "Длины y_true и y_pred должны совпадать"
  percentage_diff = np.abs((y_pred / y_true - 1) * 100)
  pred_diff = (percentage_diff >= 10).astype(int)

  share_10 = (np.sum(pred_diff) / len(pred_diff)) * 100
  return share_10

print('Доля прогнозов, отличающихся от реальных цен на авто более чем на 10% (в одну или другую сторону):')
print(f'linear regression (без стандартизации): {business_metric(y_test, y_test_lr):.1f}')
print(f'linear regression (со стандартизацией): {business_metric(y_test, y_test_trans):.1f}')
print(f'Lasso: {business_metric(y_test, y_test_lasreg):.1f}')
print(f'Ridge регрессия: {business_metric(y_test, y_test_ridgereg):.1f}')

"""1.   Стандартизация данных не повлияла на результат линейной регрессии
2.   Lasso и Ridge регрессия показали некоторое улучшение. Lasso и Ridge, будучи методами регуляризации, способны лучше справляться с переобучением и шумом в данных по сравнению с обычной линейной регрессией. Ridge дает наиболее заметное улучшение.




"""





"""# **Часть 5 (3 балла) | Реализация сервиса на FastAPI**

### **Задание 18 (3 балла)**

Cделайте с помощью FastAPI сервис, который с точки зрения пользователя реализует две функции:

1. на вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины
2. на вход подается csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах

С точки зрения реализации это означает следующее:
- средствами pydantic должен быть описан класс базового объекта
- класс с коллецией объектов
- метод post, который получает на вход один объект описанного класса
- метод post, который получает на вход коллекцию объектов описанного класса

Шаблон для сервисной части дан ниже. Код необходимо дополнить и оформить в виде отдельного .py-файла.
"""

!pip install fastapi

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return ...


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return ...

"""Протестируйте сервис на корректность работы и приложите скриншоты (см. ниже)."""



# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/data

import pickle

with open('ridge.pkl', 'wb') as file:
    pickle.dump(ridge, file)

"""# **Часть 6 (1 балл) | Оформление результатов**

### **Задание 19 (1 балл)**

**Результаты вашей работы** необходимо разместить в своем Гитхабе. Под результатами понимаем следующее:
* ``.ipynb``-ноутбук со всеми проведёнными вами экспериментами (output'ы ячеек, разумеется, сохранить)
* сохраненный дашборд в любом формате
* ``.py``-файл с реализацией сервиса
* ``.pickle``-файл с сохранёнными весами модели, коэффициентами скейлинга и прочими числовыми значениями, которые могут понадобиться для инференса
* ``.md``-файл с выводами про проделанной вами работе:
    * что было сделано
    * с какими результатами
    * что дало наибольший буст в качестве
    * что сделать не вышло и почему (это нормально, даже хорошо😀)

**За что могут быть сняты баллы в этом пункте:**
* за отсутствие ``.pickle``-файла с весами использованной модели
* за недостаточную аналитику в ``.md``-файле
* за оформление и логику кода (в определённом смысле это тоже элемент оформления решения)

**Как будет выглядет проверка всего домашнего задания?**
1. Ассистент проходит по ссылке на (**открытый**) репозиторий из Энитаска
2. Смотрит ``readme.md``:
    * пожалуйста, приложите в него же скрины работы вашего сервиса -- собирать ваши проекты довольно времязатратно, но хочется убедиться, что всё работает
    * можете в md-файл приложить ссылку на screencast с демонстрацией
3. Просматривает ноутбук с DS-частью
4. Заглядывает в код сервиса
5. Хвалит

# **Часть Благодарственная**

Надеемся, вы честно проделали все пункты, а не просто пролистали досюда. Потому что здесь награда за старания. Пожалуйста, не стоит награждать себя до того, как закончите работать над домашкой!

<details>
<summary><b>Что-то приятное</b></summary>

**Напоминаем, что нашем курсе действует система кото-бонусов** 🐈

На фото по ссылке — сэр кот кого-то из команды курса (преподаватель, помощник преподавателя, ассистенты).

Предлагаем вам угадать — чей это товарищ!

[Первый кот](https://ibb.co/XbnpCTg)

</details>
"""