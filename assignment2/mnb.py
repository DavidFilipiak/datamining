from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
import main
import numpy

def remove_stopping_words(data):
    return [word for word in data.split()]
    #return [word for word in data.split() if word not in ENGLISH_STOP_WORDS]

train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)

vectorizer = CountVectorizer(analyzer=remove_stopping_words)
train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
test_transformed = vectorizer.transform(test_data.values[:, 1])

model = MultinomialNB()
model.fit(train_transformed, list(train_data.values[:, 0]))
predictions = model.predict(test_transformed)

print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))




