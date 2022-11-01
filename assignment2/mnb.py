from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

import main
import numpy


sparsity_range = 6
chi_square_range = 10
results_array = [[0 for _ in range(chi_square_range)] for _ in range(sparsity_range)]


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=190):
    labelId = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names_out()
    topn = sorted(zip(classifier.feature_log_prob_[labelId], feature_names))[:]
    return topn

def bayes_analysis(X_train, y_train, i, j):
    # Basic model without any tuned parameters
    model = MultinomialNB()
    # Fit model
    model.fit(X_train, y_train)
    parameters = {
        'alpha': [x / 10 for x in range(1, 50)],
    }
    # Create grid serach to find the best C value
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
    grid.fit(X_train, y_train)

    #print('Best parameters: \n', grid.best_params_)
    #print('Best score: \n', grid.best_score_)

    results_array[i][j] = (grid.best_params_["alpha"], round(grid.best_score_, 3))

    best_model = grid.best_estimator_

    return best_model

def analyze(vectorizer, train_data, test_data, i, j):
    train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
    test_transformed = vectorizer.transform(test_data.values[:, 1])

    feature_names = vectorizer.get_feature_names_out()

    tfidf = TfidfTransformer()
    tfidf_transformed = tfidf.fit_transform(train_transformed)

    percent = 100 - (j * 10)
    chi = SelectPercentile(chi2, percentile=percent)
    chi_result = chi.fit_transform(train_transformed, list(train_data.values[:, 0]))
    test_transformed = chi.transform(test_transformed)

    model = bayes_analysis(chi_result, list(train_data.values[:, 0]), i, j)

    predictions = model.predict(test_transformed)

    #show_most_informative_features(feature_names, model)

    #print(classification_report(list(test_data.values[:, 0]), list(predictions)))
    #print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
    #main.calculate_metrics(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

def print_results_array(array):
    max_i, max_j, max_acc = -1, -1, 0
    print("\t\t\t", end=" ")
    for l in range(chi_square_range):
        add_text = ""
        if l % 100 != 0:
            add_text = " "
        print(f"{100 - (l * 10)}{add_text}\t\t\t", end=" ")
    print()
    for i in range(len(array)):
        add_text = ""
        if i == 0:
            add_text = " "
        print(f"{i / 100}{add_text}\t\t", end=" ")
        for j in range(len(array[i])):
            add_text = ""
            if array[i][j][1] * 100 % 10 == 0:
                add_text = " "
            print(f"{array[i][j]}{add_text}\t", end=" ")

            if array[i][j][1] > max_acc:
                max_i, max_j, max_acc = i, j, array[i][j][1]
        print()

    return max_i, max_j


#
# CODE THAT WORKS AND IS IMPLEMENTED STARTS HERE
#

def analyze_bayes(X_train, y_train, ngram):
    p = Pipeline([
        ("vect", CountVectorizer(stop_words="english", ngram_range=ngram)),
        ("chi", SelectPercentile(chi2)),
        ("mod", MultinomialNB())
    ])
    params = {
        "vect__min_df": [x / 100 for x in range(6)],
        "chi__percentile": [x for x in range(10, 100, 5)],
        "mod__alpha": [x / 10 for x in range(1, 20)],
    }

    grid = GridSearchCV(estimator=p, param_grid=params, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    return best_model, grid.best_params_["vect__min_df"], grid.best_params_["chi__percentile"], grid.best_params_["mod__alpha"]


def show_most_informative_features(feature_names, model, n=10):
    delta_log_prob = model.feature_log_prob_[1, :] - model.feature_log_prob_[0, :]

    print("Class 1"+("\t"*6)+"Class 0")
    coefs_with_fns = sorted(zip(delta_log_prob, feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    print()


train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds, scope=2)
test_data = main.load_data_in_frame(test_folds, scope=2)


print("\nUnigrams:")

estimator, min_df, percentile, alpha = analyze_bayes(list(train_data.values[:, 1]), list(train_data.values[:, 0]), (1,1))
print(f"Best min_df: {min_df}; Best chi2 percentile: {percentile}; Best alpha: {alpha}")
test_transformed = estimator.named_steps["vect"].transform(test_data.values[:, 1])
test_transformed = estimator.named_steps["chi"].transform(test_transformed)
bayes = estimator.named_steps["mod"]
predictions = bayes.predict(test_transformed)
show_most_informative_features(estimator.named_steps["vect"].get_feature_names_out(), bayes)
print(classification_report(list(test_data.values[:, 0]), list(predictions)))
print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))


print("\nBigrams:")

estimator, min_df, percentile, alpha = analyze_bayes(list(train_data.values[:, 1]), list(train_data.values[:, 0]), (2,2))
print(f"Best min_df: {min_df}; Best chi2 percentile: {percentile}; Best alpha: {alpha}")
test_transformed = estimator.named_steps["vect"].transform(test_data.values[:, 1])
test_transformed = estimator.named_steps["chi"].transform(test_transformed)
bayes = estimator.named_steps["mod"]
predictions = bayes.predict(test_transformed)
show_most_informative_features(estimator.named_steps["vect"].get_feature_names_out(), bayes)
print(classification_report(list(test_data.values[:, 0]), list(predictions)))
print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))


#
# CODE THAT WORKS AND IS IMPLEMENTED ENDS HERE
#

'''
print("\nUnigrams:")

for i in range(sparsity_range):
    sparsity = i / 100
    for j in range(chi_square_range):
        vectorizer = CountVectorizer(stop_words = "english", ngram_range=(1,1), min_df=sparsity)  #stop-words:"english"
        analyze(vectorizer, train_data, test_data, i, j)

i, j = print_results_array(results_array)
print(f"Best Sparcity: {i / 100}; Best chi-square test value: {100 - j * 10}; Best Alpha: {results_array[i][j][0]}; Best Accuracy: {results_array[i][j][1]}")

vectorizer = CountVectorizer(stop_words = "english", min_df=(i / 100), ngram_range=(1,1))
train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
test_transformed = vectorizer.transform(test_data.values[:, 1])
chi = SelectPercentile(chi2, percentile=(100 - j * 10))
chi_result = chi.fit_transform(train_transformed, list(train_data.values[:, 0]))
test_transformed = chi.transform(test_transformed)
model = MultinomialNB(alpha=results_array[i][j][0])
model.fit(chi_result, list(train_data.values[:, 0]))
predictions = model.predict(test_transformed)
show_most_informative_features(vectorizer.get_feature_names_out(), model)
print(classification_report(list(test_data.values[:, 0]), list(predictions)))
print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
#print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
#main.calculate_metrics(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))



print("\nBigrams:")

for i in range(sparsity_range):
    sparsity = i / 100
    for j in range(chi_square_range):
        vectorizer = CountVectorizer(stop_words = "english", ngram_range=(2, 2), min_df=sparsity)  #stop-words:"english"
        analyze(vectorizer, train_data, test_data, i, j)
        

i, j = print_results_array(results_array)
print(f"Best Sparcity: {i / 100}; Best chi-square test value: {100 - j * 10}; Best Alpha: {results_array[i][j][0]}; Best Accuracy: {results_array[i][j][1]}")

vectorizer = CountVectorizer(stop_words = "english", min_df=(i / 100), ngram_range=(2,2))
train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
test_transformed = vectorizer.transform(test_data.values[:, 1])
chi = SelectPercentile(chi2, percentile=(100 - j * 10))
chi_result = chi.fit_transform(train_transformed, list(train_data.values[:, 0]))
test_transformed = chi.transform(test_transformed)
model = MultinomialNB(alpha=results_array[i][j][0])
model.fit(chi_result, list(train_data.values[:, 0]))
predictions = model.predict(test_transformed)
show_most_informative_features(vectorizer.get_feature_names_out(), model)
print(classification_report(list(test_data.values[:, 0]), list(predictions)))
#print(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
#main.calculate_metrics(confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

'''