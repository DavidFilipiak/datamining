from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import main
import numpy

def show_most_informative_features(feature_names, model, n=5):
    for i in list(model.classes_):
        print("Class label: ", i)
        coefs_with_fns = sorted(zip(model.feature_log_prob_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        print()

def bayes_analysis(X_train, y_train):
    # Basic model without any tuned parameters
    model = MultinomialNB()
    # Fit model
    model.fit(X_train, y_train)
    parameters = {
        'alpha': [x / 10 for x in range(1, 50)],
    }
    # Create grid serach to find the best C value
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
    grid.fit(X_train, y_train)

    print('Best parameters: \n', grid.best_params_)
    print('Best score: \n', grid.best_score_)

    best_model = grid.best_estimator_

    return best_model

def analyze(vectorizer, train_data, test_data):
    train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
    test_transformed = vectorizer.transform(test_data.values[:, 1])

    feature_names = vectorizer.get_feature_names_out()

    tfidf = TfidfTransformer()
    tfidf_transformed = tfidf.fit_transform(train_transformed)

    chi = SelectPercentile(chi2, percentile=50)
    chi_result = chi.fit_transform(train_transformed, list(train_data.values[:, 0]))
    test_transformed = chi.transform(test_transformed)
    #print(train_transformed, chi_result)

    model = bayes_analysis(chi_result, list(train_data.values[:, 0]))
    predictions = model.predict(test_transformed)

    show_most_informative_features(feature_names, model)

    print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
    main.calculate_metrics(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))


train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)

#print("\nKeep stopping words:")
#vectorizer = CountVectorizer(min_df=0.05, ngram_range=(1, 2))
#analyze(vectorizer, train_data, test_data)

print("\nUnigrams:")
vectorizer = CountVectorizer(stop_words = "english", min_df=0.03)  #stop-words:"english"
analyze(vectorizer, train_data, test_data)


print("\nBigrams:")
vectorizer = CountVectorizer(stop_words = "english", ngram_range=(2, 2), min_df=0.03)  #stop-words:"english"
analyze(vectorizer, train_data, test_data)



