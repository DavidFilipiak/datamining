from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import main
import numpy

def keep_stopping_words(data):
    return [word for word in data.split()]

def remove_stopping_words(data):
    return [word for word in data.split() if not (word in ENGLISH_STOP_WORDS or word in ["youre", "th", "thats", "im"])]

def analyze(vectorizer, train_data, test_data):
    train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
    test_transformed = vectorizer.transform(test_data.values[:, 1])

    all_words_probs = {}
    feature_names = vectorizer.get_feature_names_out()

    tfidf = TfidfTransformer()
    tfidf_transformed = tfidf.fit_transform(train_transformed)

    for j in range(len(feature_names)):
        feature = list(feature_names)[j]
        for i in range(len(train_data.values[:, 1])):
            if feature in train_data.values[:, 1][i] and train_data.values[:, 0][i] in [0]:
                all_words_probs[j] = tfidf_transformed.data[j]

    sort = sorted(all_words_probs.items(), key=lambda item: item[1], reverse=True)
    sort = dict(sort)
    first_five = list(sort.items())[:5]

    for item in first_five:
        feature = feature_names[item[0]]
        print(f"{feature}: {item[1]}")

    #df = pd.DataFrame(tfidf_transformed[15].T.todense(), index=feature_names, columns=["importance"])
    #print(df.sort_values(by=["importance"], ascending=False).head(20))

    model = MultinomialNB()
    model.fit(train_transformed, list(train_data.values[:, 0]))
    predictions = model.predict(test_transformed)
    print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
    main.calculate_metrics(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))


train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)

print("\nKeep stopping words:")
vectorizer = CountVectorizer(min_df=0.03)
analyze(vectorizer, train_data, test_data)

print("\nRemove stopping words:")
vectorizer = CountVectorizer(analyzer=remove_stopping_words, min_df=0.03)  #stop-words:"english"
analyze(vectorizer, train_data, test_data)




