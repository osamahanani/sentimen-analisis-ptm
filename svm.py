from sympy import N, symbols, Eq, solve, log
import numpy
from flask import Flask, jsonify, request, render_template, redirect
from werkzeug.utils import secure_filename
from fileinput import filename
import uuid
from flask import request
from flask import Flask, render_template
from sklearn.metrics import classification_report, precision_score, recall_score
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
from sklearn.metrics import f1_score
import nltk
nltk.download('punkt')

ptm2 = pd.read_csv('ptm_training.csv', encoding="utf8")
ptm2.head()

ptm = ptm2[['Kategori', 'Label', 'Content']]
ptm.head()

string.punctuation

app = Flask(__name__)

#---------UPLOAD---------
UPLOAD_FOLDER = 'new/'
ALLOWED_EXTENTIONS = {'csv', 'excel'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
                              'gapapa', 'woi', 'apapun', 'eh', 'kah', 'tuh', 'apanya', 'tante', 'mas', 'suami', 'si', 'mama', 'bapak',
                              'nder', 'budhe', 'kakek', 'nenek', 'mbah', 'wow', 'yg', 'yang', 'uwu', 'di', 'guys', 'udh', 'ga', 'anjing', 'anj',
                               'anjir', 'anjingg', 'wkwk', 'aja', 'di', 'karena', 'ke', 'dari', 'nder', 'padahal', 'wkwkwkkwkwkwkwk', 'hmmm', 'aksgayqbwf', 'kalo', 'jd'])

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

ptm['filter']

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

# print("print first row matrix TF_IDF_Vec Series\n")
# print(ptm["TF_IDF_Vec"][0])

# print("\nmatrix size : ", len(ptm["TF_IDF_Vec"][0]))


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

# ------ Hessianmat -----------
#---hessianmat---
def partial(element, function):
    partial_diff = function.diff(element)
    return partial_diff


def gradient(partials):
    grad = numpy.matrix([[partials[0]], [partials[1]]])
    return grad
    
def gradient_to_zero(symbols_list, partials):
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)
    singular = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))
    return singular

def hessian(partials_second, cross_derivatives):
    hessianmat = numpy.matrix([[partials_second[0], cross_derivatives], [cross_derivatives, partials_second[1]]])
    return hessianmat
def determat(partials_second, cross_derivatives, singular, symbols_list):
    det = partials_second[0].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) * partials_second[1].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) - (cross_derivatives.subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]))**2
    return det  
def hessianmat():
    #(self.c + X1.dot(X2.T)) ** self.degree
    c, x1, x2, d = symbols('c, x1, x2, d')
    symbols_list = [c, x1, x2, d]
    function = (c+x1*x2)**d
    partials, partials_second = [], []

    for element in symbols_list:
            partial_diff = partial(element, function)
            partials.append(partial_diff)

    grad = gradient(partials)
    singular = gradient_to_zero(symbols_list, partials)

    cross_derivatives = partial(symbols_list[0], partials[1])

    for i in range(0, len(symbols_list)):
            partial_diff = partial(symbols_list[i], partials[i])
            partials_second.append(partial_diff)

    hessianmat = hessian(partials_second, cross_derivatives)
    #det = determat(partials_second, cross_derivatives, singular, symbols_list)

    #print("Hessian Matric function {0} = :\n {1}".format(function, hessianmat))
    #print("Determinant in the singular point {0} is :\n {1}".format(singular, det))
    return hessianmat

class MulticlassSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose
        self.degree = 1
        self.c = 1
        self.hmat = hessianmat()
    
    def _polynomial(self, x, y, i):
        g = (self.c+np.dot(x[i], self.coef_.T)+1)**self.degree
        g[y[i]] -= 1
        return g
    def _poly_hmat(self, x, y, i):
        d = self.degree
        c = self.c
        x1 = x[i]
        x2 = y[i]
        g = (c+np.dot(x[i], self.coef_.T)+1)**d
        hmatVal = [eval(str(self.hmat[0]).replace('[','').replace(']','').split(',')[0].split('\n')[0].strip()),
                   eval(str(self.hmat[0]).replace('[','').replace(']','').split(',')[0].split('\n')[1].strip()),
                   eval(str(self.hmat[1]).replace('[','').replace(']','').split(',')[0].split('\n')[0].strip())]
        g[y[i]] -= 1
        return [g, hmatVal]

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

                #g = self._partial_gradient(X, y, i)
                g, hmat = self._poly_hmat(X, y, i)
                
                
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
    #global clf
    #     from sklearn.datasets import load_iris

    #     iris = load_iris()
    #     X, y = iris.data, iris.target
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=3, random_state=0, verbose=1)
    X_train, x_test, y_train, y_test = train_test_split(np.array(ptm["TF_IDF_Vec"].tolist()), np.array(ptm["Kategori"].tolist()),
                                                        test_size=0.1, random_state=101)
    
    clf.fit(X_train, y_train)
    print("train:", clf.score(X_train, y_train))

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy)

@app.route('/')
def index():
    col_list = ["Kategori"]
    readagain = pd.read_csv('ptm_training.csv', usecols=col_list)

    pos = 0
    neg = 0
    netral = 0
    for row in readagain.iterrows():
        name = row[1]['Kategori']
        if name == 1:
            pos += 1
        elif name == 0:
            netral += 1
        elif name == -1:
            neg += 1

    total = pos + netral + neg
    d_pos = pos
    d_net = netral
    d_neg = neg
    pos = pos / total
    netral = netral / total
    neg = neg / total

    fa = accuracy * 100
    final = "{:.2f}".format(fa)
    positive = "{:.2f}".format(pos * 100)
    negative = "{:.2f}".format(neg * 100)
    netral = "{:.2f}".format(netral * 100)

    return render_template('index.html', accuracy=final, positive=positive,negative=negative,netral=netral,jumlahdata=total,d_pos=d_pos,d_net=d_net,d_neg=d_neg)

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/data')
def table():
    # converting csv to html
    data = pd.read_csv('ptm_training.csv')
    datas = data.values
    filter = ptm['filter']
    return render_template("data2.html", datas=datas, filter=filter)

# @app.route('/textprepro')
# def textprepro():
#     clean_msg = ptm['clean_msg']
#     msg_lower = ptm['msg_lower']
#     token = ptm['token']
#     filter = ptm['filter']

#     return render_template('textprepro.html', clean_msg=clean_msg, msg_lower=msg_lower, token=token,filter=filter)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

def render_table(file, vaccine):
    #global clf
    filename = secure_filename(file.filename)
    path = app.config['UPLOAD_FOLDER'] + str(uuid.uuid4()) + '-' + filename
    file.save(path)
    
    #data = pd.read_csv(path, encoding="UTF8", delimiter=",")

    # print('============')
    # print(data['TF_IDF_Vec'].values.tolist())
    # print('=============')

    X_train, x_test, y_train, y_test = train_test_split(np.array(ptm["TF_IDF_Vec"].tolist()), np.array(ptm["Kategori"].tolist()),
                                                        test_size=0.1, random_state=101)
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=3, random_state=0, verbose=1)
    clf.fit(X_train, y_train)
    data_upload2 = pd.read_csv(path, encoding="utf8")
    data_upload2.head()
    data_upload = data_upload2[['Content']]
    hasil_konten = data_upload2[['Content']].values.tolist()
    data_tanggal = data_upload2[['Created at']].values.tolist()
    data_upload['Content'] = data_upload['Content'].apply(remove_tweet_special)
    data_upload['Content'] = data_upload['Content'].apply(remove_number)
    data_upload['Content'] = data_upload['Content'].apply(remove_whitespace_LT)
    data_upload['Content'] = data_upload['Content'].apply(remove_whitespace_multiple)
    data_upload['Content'] = data_upload['Content'].apply(remove_singl_char)
   
    data_upload['clean_msg'] = data_upload['Content'].apply(lambda x: remove_punctuation(x))
    data_upload['msg_lower'] = data_upload['clean_msg'].apply(lambda x: x.lower())
    hasil_lower = data_upload['clean_msg'].apply(lambda x: x.lower()).tolist()
    
    # data_upload['token'] = data_upload['msg_lower'].apply(word_tokenize_wrapper)
    # data_upload['filter'] = data_upload['token'].apply(stopwords_removal)
    # hasil_token = data_upload['token'].apply(stopwords_removal).tolist()
    
    data_upload['token'] = data_upload['msg_lower'].apply(word_tokenize_wrapper)
    data_upload['filter'] = data_upload['token'].apply(stopwords_removal)
    hasil_token = data_upload['token'] = data_upload['msg_lower'].apply(word_tokenize_wrapper).tolist()

    data_upload["TF_dict"] = data_upload['filter'].apply(calc_TF)
    hasil_filter = data_upload['filter'].apply(stopwords_removal).tolist()

    tfdct = pd.concat([data_upload["TF_dict"],ptm['TF_dict']]).drop_duplicates().reset_index(drop=True)
    #data_upload["TF_dict"] = data_upload["TF_dict"].append(ptm['TF_dict'])

    DF = calc_DF(tfdct)
    n_document = len(data_upload)
    IDF = calc_IDF(n_document, DF)
    data_upload["TF-IDF_dict"] = tfdct.apply(calc_TF_IDF)
    #data_upload["TF-IDF_dict"] = data_upload["TF-IDF_dict"].append(ptm['TF-IDF_dict'])
    tfidfdct = pd.concat([data_upload["TF-IDF_dict"],ptm['TF-IDF_dict']]).drop_duplicates().reset_index(drop=True)

    sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:893]
    unique_term = [item[0] for item in sorted_DF]
    data_upload["TF_IDF_Vec"] = tfidfdct.apply(calc_TF_IDF_Vec)

    hasil = clf.predict(np.array(data_upload["TF_IDF_Vec"].tolist()))
    print('0000000000000000000000000000')
    print(hasil)
    print('0000000000000000000000000000')


    #--------------tata data----------------
    dtTabel = []
    dtGrafik = {}
    namaBulan = {}
    tPos = 0
    tNeg = 0
    tNet = 0
    tData = 0
    namaBulan['01'] = 'Januari'
    namaBulan['02'] = 'Februari'
    namaBulan['03'] = 'Maret'
    namaBulan['04'] = 'April'
    namaBulan['05'] = 'Mei'
    namaBulan['06'] = 'Juni'
    namaBulan['07'] = 'Juli'
    namaBulan['08'] = 'Agustus'
    namaBulan['09'] = 'September'
    namaBulan['10'] = 'Oktober'
    namaBulan['11'] = 'November'
    namaBulan['12'] = 'Desember'



    dtNo = 1
    for h in hasil:
        bulan = namaBulan[data_tanggal[dtNo-1][0].split(' ')[0].split('-')[1]]
        if not bulan in dtGrafik:
            dtGrafik[bulan] = [0,0,0]
        dtRow = []
        dtRow.append(dtNo)
        dtRow.append(hasil_konten[dtNo-1])
        dtRow.append(hasil_lower[dtNo-1])
        dtRow.append(hasil_token[dtNo-1])
        dtRow.append(hasil_filter[dtNo-1])
        if h==1 :
            tPos += 1
            dtGrafik[bulan][0]+=1
            dtRow.append('POSITIF')
        if h==-1 :
            tNeg += 1
            dtGrafik[bulan][1]+=1
            dtRow.append('NEGATIF')
        if h==0 :
            tNet += 1
            dtGrafik[bulan][2]+=1
            dtRow.append('NETRAL')
        tData+=1
        dtNo+=1
        # dtRow.append(data_tanggal[dtNo-1])
        dtTabel.append(dtRow)
    #--------------------------------------
    tPos = int(tPos/tData*100)
    tNeg = int(tNeg/tData*100)
    tNet = int(tNet/tData*100)

    posMax = ''
    negMax = ''
    netMax = ''
    posVal = 0
    negVal = 0
    netVal = 0

    for key, value in dtGrafik.items():
        if value[0]>posVal:
            posMax = key
            posVal = value[0]
        if value[1]>negVal:
            negMax = key
            negVal = value[1]
        if value[2]>netVal:
            netMax = key
            netVal = value[2]


    #return render_template('prediksi.html', tables=[data_upload.to_html(classes='data')], titles='', vaccine=vaccine, prediksi=hasil ,dataTabel = dtTabel )
    return render_template('prediksi.html',dataTabel = dtTabel, dataGrafik = dtGrafik, percentage = [tPos, tNet, tNeg], maxdata=[posMax, netMax, negMax])
    data.columns.values

@app.route('/table', methods=['GET', 'POST'])
def upload():
    upload = []
    if request.method == 'POST':
        vaccine = request.form['vaccine']
        f = request.files['csvfile1']
        return render_table(f, vaccine)

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

# @app.route('/grafik')
# def grafik():
#     return render_template('grafik.html')

@app.route('/pengujian', methods=['POST','GET'])
def pengujian():
    #global clf
    if request.method == 'POST':
        perbandingan = float(request.values.get('perbandingan').split(':')[1])/100.0
        X_train, x_test, y_train, y_test = train_test_split(np.array(ptm["TF_IDF_Vec"].tolist()), np.array(ptm["Kategori"].tolist()),
                                                        test_size=perbandingan, random_state=101)
        clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=3, random_state=0, verbose=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        fScore = f1_score(y_test, y_pred, average='macro')
        print('----')
        print(perbandingan);
        print(accuracy)
        print(precision)
        print(recall)
        print(fScore)
        print('----')
        return render_template('pengujian.html', acc=accuracy, pre=precision, re=recall, f=fScore)
    else:   
        return render_template('pengujian.html')

if __name__ == '__main__':
    app.run(debug=True)
