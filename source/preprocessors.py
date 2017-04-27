import json
import functools
import time
import pandas
import numpy as np

from pymystem3 import Mystem
from functools import reduce
from collections import Counter
"""
#for hotel classification
def get_mystem_decorator(func):
    mystem = Mystem()
    @functools.wraps(func)
    def new_func(string):
        return func(mystem, string)
    return new_func

def get_dictionary_decorator(func):
    with open("new_dictionary.txt") as f:
        dictionary = json.loads(f.readline())
    @functools.wraps(func)
    def new_func(string):
        return func(dictionary, string)
    return new_func

@DeprecationWarning
@get_mystem_decorator
def text_preprocess(mystem, string):
    #сюда можно добавить и прочую обработку, например использовать API яндекс.переводчика;
    #сейчас этот метод уменьшает размерность на 1505 единиц
    return " ".join(list(map(lambda st: mystem.lemmatize(st)[0], string.split())))
"""
MYSTEM = Mystem()

with open("../data/new_dictionary1.txt") as f:
    DICTIONARY = json.loads(f.readline())

REGIONS = set(region.lower() for region in pandas.read_csv("../data/regions.tsv", sep="\t", names=["id", "type", "name"])["name"])


def mystem_preprocess(string):
    return " ".join(list(map(lambda st: MYSTEM.lemmatize(st)[0], string.split())))


def dictionary_preprocess(string):
    return " ".join(list(map(lambda st: DICTIONARY[st] if st in DICTIONARY else st, string.split())))


def num_filter_preprocess(string):
    return " ".join(list(filter(lambda st: not st.isnumeric() or len(st) < 4, string.split())))


def buildup_set(df):
    length = df.shape[0]
    new_df1 = pandas.DataFrame([[df["yaHotelId"].iloc[i], " ".join(df["query"].iloc[i].strip().split()[:-1])] for i in range(length)],
                                   columns=["yaHotelId", "query"])
    new_df2 = pandas.DataFrame([[df["yaHotelId"].iloc[i], " ".join(df["query"].iloc[i].strip().split()[:-2])] for i in range(length)],
                                   columns=["yaHotelId", "query"])
    return df.append(new_df1).append(new_df2)


class Preprocessor:
    def __init__(self, funcs):
        self.pipeline = funcs
        self.times = []

    def get_quantile(self,p=0.9):
        return pandas.DataFrame(self.times).quantile(p)

    def preprocess(self, string):
        t1 = time.time()
        res = reduce(lambda x, y: y(x), [string] + self.pipeline)
        self.times.append(time.time() - t1)
        return res


#for non-hotel classification

def word_count_feature(line):
    return len(str(line).strip().split())


def isenglish_feature(line):
    latin_chars = set(list("abcdefghikjlmnopqrstvuwxyz"))
    for word in str(line).strip().split():
        for ch in word:
            if ch in latin_chars:
                return 1
    return 0


def istitle_feature(line):
    for word in str(line).strip().split():
        if word.istitle():
            return 1
    return 0


class IsWordFeature:
    def __init__(self, word):
        self.word = word

    def __call__(self, line):
        if self.word in list(map(lambda st: MYSTEM.lemmatize(st)[0], str(line).strip().split())):
            return 1
        return 0


def many_feature(line):
    for word in str(line).strip().split():
        word_information = MYSTEM.analyze(word)[0]
        if "analysis" in word_information and len(word_information["analysis"]) > 0 \
                and "gr" in word_information["analysis"][0] and "мн" in word_information["analysis"][0]["gr"]:
            return 1
    return 0


def bastard_words_feature(line):
    for word in str(line).strip().split():
        word_information = MYSTEM.analyze(word)[0]
        if "analysis" in word_information and len(word_information["analysis"]) > 0 and\
                        "qual" in word_information["analysis"][0] and word_information["analysis"][0]["qual"] == "bastard":
            return 1
    return 0


def isdigit_feature(line):
    for word in str(line).strip().split():
        if word.isnumeric():
            return 1
    return 0


def isregion_feature(line):
    for word in str(line).strip().split():
        if MYSTEM.lemmatize(word)[0] in REGIONS:
            return 1
    return 0


def isverb_feature(line):
    for word in str(line).strip().split():
        word_information = MYSTEM.analyze(word)[0]
        if "analysis" in word_information and len(word_information["analysis"]) > 0 \
                and "gr" in word_information["analysis"][0] and "V," in word_information["analysis"][0]["gr"]:
            return 1
    return 0


def isadjective_feature(line):
    for word in str(line).strip().split():
        word_information = MYSTEM.analyze(word)[0]
        if "analysis" in word_information and len(word_information["analysis"]) > 0 \
                and "gr" in word_information["analysis"][0] and "A=" in word_information["analysis"][0]["gr"]:
            return 1
    return 0


class NonHotelPreprocessor:

    def __init__(self, funcs):
        self.funcs = funcs

    def preprocess(self, lines):
        features = []
        for line in lines:
            line_features = list(map(lambda func: func(line), self.funcs))
            features.append(line_features)
        return np.array(features)





