#%%
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
import dataset
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

stemmer = SnowballStemmer("english", ignore_stopwords=False)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def training(data):
    c_v = CountVectorizer()
    stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(1, 2))
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
                #'tfidf__norm': ['l1', 'l2', None],
                #'tfidf__use_idf': (True, False),
                #'tfidf__smooth_idf': (True, False),
                #'clf__solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
                #'clf__penalty': ('l2', 'none'),
                }

    clf = Pipeline([('vect', stemmed_count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(n_jobs=-1, solver='saga')),
            ])

    # clf = GridSearchCV(clf, parameters, n_jobs=-1)
    x, y, extra = dataset.prepare_dataset(data)
    clf = clf.fit(x, y)
    # print(clf.best_score_)
    # print(clf.best_params_)
    # print(clf.cv_results_)
    return clf

def main(repo_path):
    train, dev, test = dataset.load_data(repo_path)
    clf = training(train)
    xd, yd, extra = dataset.prepare_dataset(dev)
    preds = clf.predict(xd)
    print('Dev Acc:', np.mean(preds == yd))

    xt, yt, extra = dataset.prepare_dataset(test, remove_no_labels=True)
    preds = clf.predict(xt)
    print('Test Acc:', np.mean(preds == yt))

    save_model(clf, repo_path)

    # plot_confusion_matrix(yt, preds, repo_path)

def save_model(model, repo_path):
    with open(os.path.join(repo_path, 'models/model_lr.pickl'), 'wb') as f:
        pickle.dump(model, f)
    print('saved trained model')

def load_model(repo_path):
    with open(os.path.join(repo_path, 'models/model_lr.pickl'), 'rb') as f:
        clf = pickle.load(f)
    return clf

def plot_confusion_matrix(yt, preds, repo_path):
    cm = confusion_matrix(yt, preds, normalize='pred')
    print(cm)
    labels = ['contradiction', 'entailment', 'neutral']
    import seaborn as sn
    plt.figure(figsize=(20, 10))
    sn.heatmap(cm, annot=True, cbar=True, xticklabels=labels, yticklabels=labels) # font size
    plt.savefig(os.path.join(repo_path, 'img/cm_lr.png'))
    # plt.show()

if __name__ == '__main__':
    repo_path = os.path.dirname(os.path.abspath(__file__))
    main(repo_path)