##удаление выбросов
def drop_elements(df, col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    upper_array = np.where(df[col] >= upper)[0]
    lower_array = np.where(df[col] <= lower)[0]

    df.drop(index=upper_array, inplace=True)
    df.drop(index=lower_array, inplace=True)

    return df

##heatmap для категориальных признаков
import pandas as pd
import seaborn as sns

# Создаем набор данных с категориальным признаком
data = {'Animal': ['Dog', 'Cat', 'Dog', 'Fish', 'Fish', 'Dog']}
df = pd.DataFrame(data)

# Используем метод value_counts() для подсчета частотности каждой категории
freq_table = df['Animal'].value_counts()

# Преобразуем таблицу частотности в формат DataFrame
df_freq_table = pd.DataFrame({'Animal': freq_table.index, 'Count': freq_table.values})

# Создаем тепловую карту на основе таблицы частотности
sns.heatmap(df_freq_table.pivot(index='Animal', columns='Count', values='Count'), annot=True, cmap='YlGnBu')

