import pickle
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree
from keras.models import load_model
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():

    model_tree = load_models("D:\Machine_Learning\models\ModelDecisionTreeClassifier.pkl")
    model_bagging = load_models("D:\Machine_Learning\models\ModelBaggingClassifier.pkl")
    model_NN = load_model('D:\Machine_Learning\models\ModelNN_Classifier.h5')

    data_not_processed = load_not_processed_data("D:\Machine_Learning\data\card_transdata.csv")

    data_pre_processed = load_test_data("D:\Machine_Learning\data\pre_card_transdata.csv")

    data_processed = load_test_data("D:\Machine_Learning\data\pred_card_transdata.csv")

###PIPELINE
    pipeline = Pipeline([('scaler', StandardScaler())])
    X = data_pre_processed.drop(columns="fraud", axis=1)
    X_transformed = pipeline.fit_transform(X)
###PIPELINE

    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Визуализация данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.header("Описание задачи")
        st.markdown("""По мере развития цифровых платежей усовершенствуются и мошеннические схемы, которые очень распространены как для платежей с использованием кредитной карты, так и без нее.  
                        По предоставленным характеристикам необходимо определить, была ли проведена мошенническая транзакция.""")

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:

* ***distance_from_home*** – расстояние до места, где произошла транзакция,
* ***distance_from_last_transaction*** – время с момента совершения последней транзакции,
* ***ratio_to_median_purchase_price*** – отношение покупной цены сделки к медианной цене покупки,
* ***repeat_retailer*** – транзакция произошла от одного и того же продавца,
* ***used_chip*** – транзакция произошла с использованием кредитной карты,
* ***used_pin_number*** – при совершении транзакции использовался пин-код,
* ***online_order*** – транзакция является онлайн-заказом,
* ***fraud*** – транзакция является мошеннической.


К вещественным признакам относятся:
* ***distance_from_home***,
* ***distance_from_last_transaction***,
* ***ratio_to_median_purchase_price***.


К бинарным признакам относятся:
* ***repeat_retailer***, где 1 - транзакция произошла от одного и того же продавца, 0 - от разных продавцов;
* ***used_chip***, где 1 - транзакция произошла с использованием кредитной карты, 0 - без использования кредитной карты;
* ***used_pin_number***, где 1 - при совершении транзакции использовался пин-код, 0 - не использовался пин-код;
* ***online_order***, где 1 - транзакция является онлайн-заказом, 0 - не является онлайн-заказом;
* ***fraud***, где 1 - транзакция является мошеннической, 0 - транзакция не является мошеннической.""")

        st.write("Количество различных меток, которые принимает целевой признак ***fraud***, равно двум (0 и 1). Следовательно, рассматривается задача бинарной классификации.")
        st.write("*Информация о данных*")
        st.dataframe(data_not_processed.describe())

        st.write("*Таблица данных*")
        st.dataframe(data_not_processed)

        st.header("Предобработка данных")
        st.write("Для получения наилучших результатов при построении моделей для решения задач машинного обучения необходимо произвести предобработку полученных данных.")

        st.write(":green[***Дисбаланс классов***]")
        st.write("Первая проблема, определенная при анализе данных, - дисбаланс классов. Количество объектов, принадлежащих нулевому классу, - ", data_not_processed["fraud"].value_counts()[0],
                 ". Количество объектов, принадлежащих первому классу, - ", data_not_processed["fraud"].value_counts()[1], ".")
        st.write("Для решения проблемы дисбаланса были удалены объекты мажоритарного класса(*down sampling*). После чего выборка сократилась до", data_processed.shape[0],
                 ". При этом количество объектов, относящихся как к нулевому, так и к первому классу, стало равным", data_processed["fraud"].value_counts()[0], ".")

        st.markdown(":green[***Масштабирование данных***]")
        st.write("При работе с данными модель подбирает соответствующие каждому признаку веса, от которых зависит результат предсказания. В ситуации, когда диапазон значений признаков отличается друг от друга на порядки, модель может начать ошибаться."
                 " Эта ситуация связана с тем, что при обучении алгоритма главный приоритет, и, соответственно, большие веса приобретают те признаки, значения которых во много раз больше остальных. Следовательно, столбцы со значениями из малого диапозона, получают меньшие веса или зануляются.")
        st.write("Для того чтобы степень важности признаков не зависела только от значений, необходимо проводить масштабирование.")
        st.write("Как видно из фрагмента вышеприведенной таблицы, значения, принимаемые признаками, отличаются друг от друга в десять и более раз. Согласуем признаки, переведя их значения в меньший масштаб.")
        st.write("C помощью StandardScaler из библиотеки sklearn данные были стандартизированы, приведены к виду со средним значением, равным нулю, и стандартным отклонением, равным единице.")

        st.write("*Информация о данных*")
        st.dataframe(data_processed.describe())

        st.write("*Преобразованная таблица данных*")
        st.dataframe(data_processed)

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        request = st.selectbox(
            "Выберите запрос",
            ["Пять предсказанных значений", "Индивидуальный запрос"]
        )

        if request == "Индивидуальный запрос":
            st.title("Индивидуальный запрос")
            choose = st.selectbox(
            "Выберите запрос",
            ["DecisionTreeClassifier", "BaggingClassifier", "NN_Classifier"])
            if choose == "DecisionTreeClassifier":

                one_object_prediction(model_tree, pipeline)

            if choose == "BaggingClassifier":

                one_object_prediction(model_bagging, pipeline)

            if choose == "NN_Classifier":

                one_object_prediction(model_NN, pipeline)

        elif request == "Пять предсказанных значений":

            choose = st.selectbox(
                "Выберите запрос",
                ["DecisionTreeClassifier", "BaggingClassifier", "NN_Classifier"])

            if choose == "DecisionTreeClassifier":

                predictions_from_file(model_tree, pipeline)

            elif choose == "BaggingClassifier":

                predictions_from_file(model_bagging, pipeline)

            elif choose == "NN_Classifier":

                predictions_from_file(model_NN, pipeline)

    elif page == "Визуализация данных":
        st.title("Визуализация данных")

        st.write("* **Histograms**. Гистограммы показывают, как изменялось количество объектов в каждом классе до и после решения проблемы дисбаланса.")
        fig_class = plt.figure(figsize=(8, 8))
        plt.subplot(1, 3, 1)
        data_not_processed['fraud'].plot.hist()
        plt.xticks([0, 1])
        plt.subplot(1, 3, 3)
        data_processed['fraud'].plot.hist()
        plt.xticks([0, 1])
        st.pyplot(fig_class)

        st.write("* **Heatmap**. Диаграмма показывает корреляцию между всеми парами признаков в виде цветовой карты.")
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data_processed.drop(['fraud'], axis=1).corr(),
                    linewidth=0.3, vmax=1.0, square=True, linecolor='black', annot=True)
        st.pyplot(fig)

        st.write("* **BoxPlot**. Графики показывают распределение числовых данных с возможностью идентифицикации выбросов.")

        fig, ax = plt.subplots(1, 3, figsize=(10, 7))
        ax[0].boxplot(data_processed["distance_from_home"])
        ax[0].set_title("distance_from_home")

        ax[1].boxplot(data_processed["distance_from_last_transaction"])
        ax[1].set_title("distance_from_last_transaction")

        ax[2].boxplot(data_processed["ratio_to_median_purchase_price"])
        ax[2].set_title("ratio_to_median_purchase_price")
        st.pyplot(fig)

        st.write("* **DecisionTreeClassifier**. Дерево решений, в котором каждый узел представляет признак, каждая ветвь представляет возможное значение признака, а каждый лист представляет конечную метку класса.")
        fig = plt.figure(figsize=(25, 25))
        tree.plot_tree(model_tree, feature_names=data_processed.columns, filled=True)
        st.pyplot(fig)


def predictions_from_file(model, pipeline):
    uploaded_file = st.file_uploader("Выберите файл CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        st.write(df)
        st.header("Первые 5 предсказанных значений")
        first_5_test = df.iloc[:5, :-1]
        st.dataframe(df.iloc[:5, :])
        test = pipeline.transform(first_5_test)
        predict = model.predict(test)
        for i, item in enumerate(predict):
            st.write(i, f"{int(item)}")
    else:
        print("Ошибка открытия файла")


def one_object_prediction(model, pipeline):

    data = add_fields()

    if st.button('Предсказать'):

        data = np.array(data).reshape((1, -1))
        data = pipeline.transform(data)
        predict = model.predict(data)
        st.write("Предсказанное значение:", int(predict))


def add_fields():

    distance_from_home = st.number_input("Расстояние от дома")
    distance_from_last_transaction = st.number_input("Время с момента совершения последней транзакции")
    ratio_to_median_purchase_price = st.number_input("Отношение покупной цены сделки к медианной цене покупки")
    repeat_retailer = st.selectbox("Tранзакция произошла от одного и того же продавца", [0, 1])
    repeat_retailer = int(repeat_retailer)
    used_chip = st.selectbox("Tранзакция произошла при помощи кредитной карты", [0, 1])
    used_chip = int(used_chip)
    used_pin_number = st.selectbox("При совершении транзакции использовался пин-код", [0, 1])
    used_pin_number = int(used_pin_number)
    online_order = st.selectbox("Транзакция является онлайн-заказом", [0, 1])
    online_order = int(online_order)
    return [distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer,
            used_chip, used_pin_number, online_order]

@st.cache_data
def load_models(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df


@st.cache_data
def load_not_processed_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


if __name__ == '__main__':
    main()
