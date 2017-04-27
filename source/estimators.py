import math
import numpy as np
import time
import pandas
import shutil
import os
import filters
import whoosh

from functools import reduce
from sklearn.neighbors import LSHForest
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from collections import OrderedDict
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.analysis import SpaceSeparatedTokenizer
from whoosh.analysis import LowercaseFilter
from whoosh.writing import BufferedWriter
from whoosh.index import LockError
from whoosh.qparser import QueryParser
from whoosh import qparser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from whoosh.scoring import TF_IDF


def epan_parzen_function(r):
    return 3/4*(1 - r*r)


def quart_parzen_function(r):
    return 15/16*(1 - r*r)*(1 - r*r)


def trian_parzen_function(r):
    return 1 - abs(r)


def gauss_parzen_function(r):
    return (2 * math.pi)**(-0.5)*math.exp(-0.5*r*r)


class LSHNearestNeighbors(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=10, n_candidates=50, n_neighbors=9, mode="most_common", parzen_func=epan_parzen_function, answers="one"):
        self.n_estimator = n_estimators
        self.n_candidates = n_candidates
        self.n_neighbors = n_neighbors
        self.parzen_func = parzen_func
        assert mode in ["most_common", "parzen_window", "dann", "voting_parzen_window"]
        self.mode = mode
        assert answers in ["one", "many"]
        self.answers = answers

    def fit(self, X, y):
        self.INNER_NEIGHBORS = 50
        if self.mode == "dann":
            self.lshforest_ = LSHForest(n_estimators=self.n_estimator, n_candidates=self.n_candidates,
                                    n_neighbors=self.INNER_NEIGHBORS)
        else:
            self.lshforest_ = LSHForest(n_estimators=self.n_estimator, n_candidates=self.n_candidates,
                                    n_neighbors=self.n_neighbors)
        self.lshforest_.fit(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def get_neighbors(self, x):
        return [self.y_[mark] for mark in self.lshforest_.kneighbors(x)[1].tolist()[0]]

    def get_neighbors_n_dists(self, x):
        res = self.lshforest_.kneighbors(x)
        return [(self.y_[res[1][0, i]], res[0][0, i]) for i in range(self.n_neighbors)]

    def get_neighbors_n_dists_n_coordinates(self, x):
        res = self.lshforest_.kneighbors(x)
        return [(self.y_[res[1][0, i]], res[0][0, i], self.X_[res[1][0, i], :]) for i in range(self.n_neighbors)]

    def get_quantile(self,p=0.9):
        return pandas.DataFrame(self.times).quantile(p)

    def predict(self, X):
        self.times = []
        y_pred = []
        if self.mode == "most_common":
            for x in X:
                t1 = time.time()
                kneighbors_y = self.get_neighbors(x)
                self.times.append(time.time() - 1)
                y_pred.append(Counter(kneighbors_y).most_common(1)[0][0])
        elif self.mode == "parzen_window":
            for x in X:
                t1 = time.time()
                k_n_d = self.get_neighbors_n_dists(x)
                #print("k_n_d", k_n_d)
                rating = OrderedDict()
                max_ind = -1
                max_rate = -1
                if self.answers == "one":
                    for i in range(self.n_neighbors):
                        if k_n_d[i][0] not in rating.keys():
                            rating[k_n_d[i][0]] = self.parzen_func(k_n_d[i][1]/(k_n_d[self.n_neighbors - 1][1] + 0.0001))
                        else:
                            rating[k_n_d[i][0]] += self.parzen_func(k_n_d[i][1]/(k_n_d[self.n_neighbors - 1][1] + 0.0001))
                        if rating[k_n_d[i][0]] > max_rate:
                            max_rate = rating[k_n_d[i][0]]
                            max_ind = k_n_d[i][0]
                    self.times.append(time.time() - t1)
                    y_pred.append(max_ind)
                elif self.answers == "many":
                    for i in range(self.n_neighbors):
                        if k_n_d[i][0] not in rating.keys():
                            rating[k_n_d[i][0]] = self.parzen_func(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1])
                        else:
                            rating[k_n_d[i][0]] += self.parzen_func(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1])
                    sum_weights = sum(rating[k] for k in rating)
                    for k in rating:
                        rating[k] /= sum_weights
                    self.times.append(time.time() - t1)
                    y_pred.append(rating)
        elif self.mode == "voting_parzen_window":
            for x in X:
                t1 = time.time()
                k_n_d = self.get_neighbors_n_dists(x)
                rating = OrderedDict()
                parzen_functions = [epan_parzen_function, quart_parzen_function, trian_parzen_function, gauss_parzen_function]
                max_ind = -1
                max_rate = -1
                if self.answers == "one":
                    for i in range(self.n_neighbors):
                        if k_n_d[i][0] not in rating.keys():
                            rating[k_n_d[i][0]] = sum(map(lambda func: func(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1]), parzen_functions))
                        else:
                            rating[k_n_d[i][0]] += sum(map(lambda func: func(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1]), parzen_functions))
                        if rating[k_n_d[i][0]] > max_rate:
                            max_rate = rating[k_n_d[i][0]]
                            max_ind = k_n_d[i][0]
                    self.times.append(time.time() - t1)
                    y_pred.append(max_ind)
                elif self.answers == "many":
                    for i in range(self.n_neighbors):
                        if k_n_d[i][0] not in rating.keys():
                            rating[k_n_d[i][0]] = epan_parzen_function(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1])
                        else:
                            rating[k_n_d[i][0]] += epan_parzen_function(k_n_d[i][1]/k_n_d[self.n_neighbors - 1][1])
                    sum_weights = sum(rating[k] for k in rating)
                    for k in rating:
                        rating[k] /= sum_weights
                    self.times.append(time.time() - t1)
                    y_pred.append(rating)

        """
        elif self.mode == "dann":
            for x in X:
                k_n_d_n_c = self.get_neighbors_n_dists_n_coordinates(x)
                sigma = np.eye(self.INNER_NEIGHBORS)
                sqrt_sigma = np.linalg.sqrtm(sigma)
                d = [np.linalg.norm(np.dot(sqrt_sigma,x - k_n_d_n_c[i][2])) for i in range(len(k_n_d_n_c))]
                h = max(d)
                w = [(1 - (d[i]/h)**3)**3 for i in range(self.INNER_NEIGHBORS)]
                pi = OrderedDict()
                for key in [k_n_d_n_c[i][0] for i in range(len())]:
                    pi[key] = sum(w[i] for i in range(len(w)) if k_n_d_n_c[i][0] == key)/sum(w)
                B = sum(pi[i] * np.dot((k_n_d_n_c[i][2] - x).reshape(len(k_n_d_n_c[i][2], 1)),
                                       (k_n_d_n_c[i][2] - x).reshape(1, len(k_n_d_n_c[i][2))) for i in range(len(k_n_d_n_c)))
        """
        return y_pred



class Bayes_Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, num_answers=1, answers="one"):
        self.alpha = alpha
        assert answers in ["one", "many"]
        self.answers = answers
        self.num_answers = num_answers

    def iter_minichunks(self, X, y, chunk_size):
        num_chunks = X.shape[0]//chunk_size
        for i in range(1, num_chunks):
            yield X[(i - 1)*chunk_size : i*chunk_size, :], y[(i - 1)*chunk_size : i*chunk_size]

    def fit(self, X, y):
        self.inner_clf_ = MultinomialNB(alpha=self.alpha)
        classes = np.unique(y)
        chunkerator = self.iter_minichunks(X, y, chunk_size=100)
        for X_chunk, y_chunk in chunkerator:
            print(X_chunk.shape, y_chunk.shape)
            print(type(X_chunk), type(y_chunk))
            self.inner_clf_.partial_fit(X_chunk, y_chunk, classes=classes)

    def predict(self, X, y):
        if self.answers == "one":
            return list(self.inner_clf_.predict(X))
        elif self.answers == "many":
            y_pred = []
            for x in X:
                probs = list(self.inner_clf_.predict_proba(x))[0]
                probs_n_class = sorted([(self.classes_[i], probs[i]) for i in range(len(probs))], key=lambda item: item[1], reverse=True)[:self.num_answers]
                sum_probs = sum(probs_n_class[i][1] for i in range(len(probs_n_class)))
                normed_probs_n_class = [(probs_n_class[i][0], probs_n_class[i][1]/sum_probs) for i in range(len(probs_n_class))]
                rating = OrderedDict(normed_probs_n_class)
                y_pred.append(rating)
            return y_pred


class MixedClassifier:

    def __init__(self, clfs, weights):
        assert len(clfs) == len(weights)
        for i in range(len(clfs)):
            assert clfs[i].answers == "many"
        self.clfs = clfs
        self.weights = weights

    def predict(self, X):
        y_pred = []
        for x in X:
            max_rate = -1
            max_ind = -1
            total_rating = OrderedDict()
            for i in range(len(self.clfs)):
                rating = self.clfs[i].predict(x)
                for k in rating:
                    if k not in total_rating:
                        total_rating[k] = rating[k]*self.weights[i]
                    else:
                        total_rating[k] += rating[k]*self.weights[i]
                    if total_rating[k] > max_rate:
                        max_rate = total_rating[k]
                        max_ind = k
            y_pred.append(max_ind)
        return y_pred


class WhooshClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, df):
        schema = Schema(content=TEXT(stored=True, analyzer=SpaceSeparatedTokenizer() | filters.MystemFilter() | filters.TranslateFilter()), id=ID(stored=True))
        if os.path.exists("indexdir"):
            shutil.rmtree("indexdir")
        os.mkdir("indexdir")
        self.ix = create_in("indexdir", schema)
        writer = self.ix.writer()
        for d in df.iterrows():
            writer.add_document(content=d[1]["query"], id=str(d[1]["yaHotelId"]))
        writer.commit()
        return self

    def predict(self, df):
        parser = QueryParser("content", schema=self.ix.schema, group=qparser.OrGroup)
        y = []
        with self.ix.searcher(weighting=whoosh.scoring.Frequency) as searcher:
            for q in df.iterrows():
                print("start")
                t1 = time.time()
                query = parser.parse(q[1]["query"])
                results = [(int(res["id"]), res.score) for res in searcher.search(query, limit=10)]
                print("pretty finish", time.time() - t1)
                rating = OrderedDict()
                max_rate = -1
                max_ind = -1
                for i in range(len(results)):
                    if results[i][0] in rating:
                        rating[results[i][0]] += epan_parzen_function(results[i][1])
                    else:
                        rating[results[i][0]] = epan_parzen_function(results[i][1])
                    if rating[results[i][0]] > max_rate:
                        max_rate = rating[results[i][0]]
                        max_ind = results[i][0]
                print("finish", time.time() - t1)
                y.append(max_ind)
        return y


if __name__ == "__main__":
    import preprocessors

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import GridSearchCV

    path = "query.yaHotelId.showInTop.sure.final.tsv"
    df = pandas.read_csv(path, sep="\t")
    print("Изначальная размерность данных:", df.shape, ";", "Количество отелей:", len(df["yaHotelId"].unique()))

    sure_df = df[df["sure"]]
    filtered_values = [value[0] for value in sure_df["yaHotelId"].value_counts().iteritems() if value[1] >= 5]
    filtered_df = sure_df[sure_df["yaHotelId"].isin(filtered_values)]
    print("Получившаяся размерность данных:", filtered_df.shape, ";", "Количество отелей:", len(filtered_df["yaHotelId"].unique()))

    df_train, df_test = train_test_split(filtered_df, test_size=0.01)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    ids = set(df_train["yaHotelId"])
    df_hotels = pandas.read_csv("hotel_all.csv")
    df_hotels = df_hotels[df_hotels["id"].isin(ids)]
    df_hotels.rename(columns={"id": "yaHotelId", "all": "query"}, inplace=True)
    df_hotels.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
    df_hotels.dropna(axis=0, how="any", inplace=True)

    df_tr = df_train.append(df_hotels)
    df_tr.reset_index(drop=True, inplace=True)

    prep = preprocessors.Preprocessor([preprocessors.num_filter_preprocess, preprocessors.mystem_preprocess, preprocessors.dictionary_preprocess])
    vectorizer = TfidfVectorizer(preprocessor=prep.preprocess)
    y_train = np.array(df_tr["yaHotelId"])
    X_train = vectorizer.fit_transform(df_tr["query"])

    y_test = np.array(df_test["yaHotelId"])
    X_test = vectorizer.transform(df_test["query"])

    """
    param_grid = [{"n_estimators": [15], "n_candidates": [200], "n_neighbors": [5, 9, 15], "mode": ["parzen_window"],
                   "parzen_func": [epan_parzen_function, quart_parzen_function, trian_parzen_function, gauss_parzen_function],
                   },
                  {"n_estimators": [15], "n_candidates": [200], "n_neighbors": [5, 9, 15], "mode": ["voting_parzen_window"]}]

    clf = LSHNearestNeighbors()
    cv = GridSearchCV(clf, param_grid=param_grid, verbose=5)
    cv.fit(X_train, y_train)
    print(cv.best_score_)
    print(cv.best_params_)

    """
    clf = LSHNearestNeighbors(n_estimators=15, n_candidates=200, n_neighbors=9, mode="parzen_window", parzen_func=epan_parzen_function)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))

    print("precision:", precision_score(y_test, y_pred, average="weighted"))

    print("recall:", recall_score(y_test, y_pred, average="weighted"))
