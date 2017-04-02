import json
import functools
import time
import pandas

from pymystem3 import Mystem
from functools import reduce


def get_mystem_decorator(func):
    mystem = Mystem()
    @functools.wraps(func)
    def new_func(string):
        return func(mystem, string)
    return new_func

def get_dictionary_decorator(func):
    with open("dictionary.txt") as f:
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

@get_mystem_decorator
def mystem_preprocess(mystem, string):
    return " ".join(list(map(lambda st: mystem.lemmatize(st)[0], string.split())))

@get_dictionary_decorator
def dictionary_preprocess(dictionary, string):
    return " ".join(list(map(lambda st: dictionary[st] if st in dictionary else st, string.split())))


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


