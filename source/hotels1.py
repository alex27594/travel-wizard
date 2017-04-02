import pandas
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score

from estimators import LSHNearestNeighbors
from preprocessors import text_preprocess


if __name__ == "__main__":
    df = pandas.read_csv("/media/alexander/b32bf4b4-8724-4107-9d19-abf6615c2f60/alexander/HELP_FILE/query.yaHotelId.showInTop.sure.final.tsv", sep="\t")
    print("Изначальная размерность данных:", df.shape,";", "Количество отелей:", len(df["yaHotelId"].unique()))
    sure_df = df[df["sure"]]
    print(sure_df.shape)
    filtered_values = [value[0] for value in sure_df["yaHotelId"].value_counts().iteritems() if value[1] >= 5]
    filtered_df = sure_df[sure_df["yaHotelId"].isin(filtered_values)]
    print("Получившаяся размерность данных:", filtered_df.shape, ";", "Количество отелей:", len(filtered_df["yaHotelId"].unique()))

    vectorizer = TfidfVectorizer(preprocessor=text_preprocess)
    y = np.array(filtered_df["yaHotelId"])
    X = vectorizer.fit_transform(filtered_df["query"])
    print("X shape:", X.shape)

    scaler = MaxAbsScaler()
    scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = LSHNearestNeighbors(n_estimators=10, n_candidates=100, n_neighbors=9, mode="parzen window")
    clf.fit(X_train, y_train)
    t1 = time.time()
    y_pred = clf.predict(X_test)
    t2 = time.time() - t1
    print("delta time:", t2)
    print("mean time for one query:", t2/X_test.shape[0])
    print("accuracy:", accuracy_score(y_test, y_pred))