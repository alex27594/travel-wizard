import json
import requests
import pandas

from pymystem3 import Mystem

API_KEY = "api_key"

if __name__ == "__main__":
    not_translated = []
    dictionary = {}
    print(len(dictionary.keys()))
    m = Mystem()
    df = pandas.read_csv("/media/alexander/b32bf4b4-8724-4107-9d19-abf6615c2f60/alexander/HELP_FILE/query.yaHotelId.showInTop.sure.final.tsv", sep="\t")
    df_size = len(df["query"])
    k = 1
    for line in df["query"]:
        print(k, "query from", df_size)
        k += 1
        for word in line.strip().split():
            lema_word = m.lemmatize(word)[0]
            if dictionary.get(lema_word) is None:
                params = {"key": API_KEY, "text": lema_word, "lang": "ru-en"}
                try:
                    r = requests.get("https://translate.yandex.net/api/v1.5/tr.json/translate", params=params)
                    r_json = r.json()
                    trans_word = r_json["text"][0]
                    if r_json["code"] != 200:
                        print("ERROR", r_json["code"])
                        not_translated.append(lema_word)
                        continue
                except Exception as exc:
                    print("ERROR")
                    not_translated.append(lema_word)
                    continue
                if (len(trans_word.split()) > 1):
                    trans_word = "".join(trans_word.split())
                trans_word = trans_word.lower()
                dictionary[lema_word] = trans_word
    with open("data/dictionary1.txt", "w")as dict_f:
        dict_f.write(json.dumps(dictionary))
    with open("not_translate.txt", "w") as error_f:
        error_f.write(json.dumps(not_translated))
    print("Size of dictionary", len(dictionary.keys()))
    print(dictionary)



