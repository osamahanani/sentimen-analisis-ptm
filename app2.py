from sympy import symbols, Eq, solve, log
import numpy
from flask import Flask, render_template
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('punkt')

ptm2 = pd.read_csv('ptm_training.csv', encoding="ISO-8859-1")
ptm2.head()

ptm = ptm2[['Kategori', 'Label', 'Content']]
ptm.head()

string.punctuation

# ------ Case Folding ---------


def remove_tweet_special(content):
    # remove tab, new line, ans back slice
    content = content.replace('\\t', " ").replace(
        '\\n', " ").replace('\\u', " ").replace('\\', "")
    # remove non ASCII (emoticon, chinese word, .etc)
    content = content.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    content = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", content).split())
    # remove incomplete URL
    return content.replace("http://", " ").replace("https://", " ")


ptm['Content'] = ptm['Content'].apply(remove_tweet_special)

# remove number


def remove_number(content):
    return re.sub(r"\d+", "", content)


ptm['Content'] = ptm['Content'].apply(remove_number)

# remove whitespace leading & trailing


def remove_whitespace_LT(text):
    return text.strip()


ptm['Content'] = ptm['Content'].apply(remove_whitespace_LT)

# remove multiple whitespace into single whitespace


def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)


ptm['Content'] = ptm['Content'].apply(remove_whitespace_multiple)

# remove single char


def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


ptm['Content'] = ptm['Content'].apply(remove_singl_char)

# defining the function to remove punctuation


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# storing the puntuation free text
ptm['clean_msg'] = ptm['Content'].apply(lambda x: remove_punctuation(x))
ptm.head()

ptm['msg_lower'] = ptm['clean_msg'].apply(lambda x: x.lower())
ptm.head()

# ------ Tokenizing ---------

# NLTK word rokenize


def word_tokenize_wrapper(content):
    return word_tokenize(content)


ptm['token'] = ptm['msg_lower'].apply(word_tokenize_wrapper)
ptm.head()

# ------ Stopword removal ---------
factory = StopWordRemoverFactory()

Sastrawi_StopWords_id = factory.get_stop_words()

# ---------------------------- manualy add stopword  ------------------------------------
# tambahan
Sastrawi_StopWords_id.extend(['ibu', 'ayah', 'adik', 'kakak', 'nya', 'yah', 'sih', 'oke', 'kak', 'deh', 'mah', 'an', 'ku', 'mu', 'iya', 'apa',
                              'gapapa', 'akupun', 'apapun', 'eh', 'kah', 'mengada', 'apanya', 'tante', 'mas', 'suami', 'si', 'mama', 'bapak',
                              'nder', 'budhe', 'kakek', 'nenek', 'mbah', 'wow', 'yg', 'yang', 'uwu', 'di', 'guys', 'udh', 'ga'])

# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("stopwords.txt", names=[" "], header=None)

# convert stopword string to list & append additional stopword
Sastrawi_StopWords_id.extend(txt_stopword[" "][0].split(' '))

# ---------------------------------------------------------------------------------------

Sastrawi_StopWords_id = set(Sastrawi_StopWords_id)


def stopwords_removal(words):
    return [word for word in words if word not in Sastrawi_StopWords_id]


ptm['filter'] = ptm['token'].apply(stopwords_removal)

ptm.head()


def calc_TF(document):
    # Counts the number of times the word appears in review
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # Computes tf for each word
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict


ptm["TF_dict"] = ptm['filter'].apply(calc_TF)

ptm["TF_dict"].head()


def calc_DF(tfDict):
    count_DF = {}
    # Run through each document's tf dictionary and increment countDict's (term, doc) pair
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF


DF = calc_DF(ptm["TF_dict"])

# menghitung IDF
n_document = len(ptm)


def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict


# Stores the idf dictionary
IDF = calc_IDF(n_document, DF)

# calc TF-IDF


def calc_TF_IDF(TF):
    TF_IDF_Dict = {}
    # For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict


# Stores the TF-IDF Series
ptm["TF-IDF_dict"] = ptm["TF_dict"].apply(calc_TF_IDF)
ptm["TF-IDF_dict"].head()

# sort descending by value for DF dictionary
sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:893]

# Create a list of unique words from sorted dictionay `sorted_DF`
unique_term = [item[0] for item in sorted_DF]


def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)

    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector


ptm["TF_IDF_Vec"] = ptm["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

print("print first row matrix TF_IDF_Vec Series\n")
print(ptm["TF_IDF_Vec"][0])

print("\nmatrix size : ", len(ptm["TF_IDF_Vec"][0]))


def projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]

        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))

        norms = np.sqrt(np.sum(X ** 2, axis=1))

        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]

                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                delta = self._solve_subproblem(g, y, norms, i)

                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                pass
                #print("iter", it + 1, "violation", vratio)

            if vratio < self.tol:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)


if __name__ == '__main__':
    #     from sklearn.datasets import load_iris

    #     iris = load_iris()
    #     X, y = iris.data, iris.target

    X_train, x_test, y_train, y_test = train_test_split(np.array(ptm["TF_IDF_Vec"].tolist()), np.array(ptm["Kategori"].tolist()),
                                                        test_size=0.1, random_state=101)
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=3, random_state=0, verbose=1)
    clf.fit(X_train, y_train)
    print("train:", clf.score(X_train, y_train))

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy)


app = Flask(__name__)


@app.route('/')
def index():
    fa = accuracy * 100
    final = "{:.2%}".format(fa)
    positive = 10
    negative = 15
    netral = 75
    return render_template('index.html', accuracy=fa, positive=positive, negative=negative, netral=netral)

@app.route('/data')
def data():
    return render_template('data2.html', z=accuracy)

@app.route('/pengujian')
def pengujian():
    return render_template('pengujian.html')

if __name__ == '__main__':
    app.run(debug=True)
