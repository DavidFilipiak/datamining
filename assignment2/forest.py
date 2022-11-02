import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

def forest_analysis(X_train, y_train):

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    parameters = {'n_estimators': [200,400,600,800],
                  #'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                  'criterion': ['gini', 'entropy']}

    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
    grid.fit(X_train,y_train)

    print('Best parameters: \n', grid.best_params_)
    print('Best score: \n', grid.best_score_)

    best_model = grid.best_estimator_

    return best_model

def show_most_informative_features(feature_names, model, X_test, y_test, n=10):
    importances = permutation_importance(model, X_test, y_test)

    print("Class 1"+("\t"*6)+"Class 0")
    coefs_with_fns = sorted(zip(importances, feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    print()

# Data
train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)

# Unigram
print("\nUnigram Model Analysis: \n")
vectorizer = CountVectorizer(analyzer="word", stop_words = "english")
train_unigram = vectorizer.fit_transform(train_data.values[:, 1])
test_unigram = vectorizer.transform(test_data.values[:, 1])

unigram_model = forest_analysis(train_unigram, list(train_data.values[:, 0]))
#show_most_informative_features(vectorizer.get_feature_names_out(), unigram_model, test_data.values[:, 1],test_data.values[:, 0])


predictions = (unigram_model.predict(test_unigram))
print(predictions)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

# Bigram
print("\nUnigram and Bigram Model Analysis: \n")
vectorizer2 = CountVectorizer(analyzer="word", stop_words = "english", ngram_range=(1, 2))
train_bigram = vectorizer2.fit_transform(train_data.values[:, 1])
test_bigram = vectorizer2.transform(test_data.values[:, 1])

bigram_model = forest_analysis(train_bigram, list(train_data.values[:, 0]))
#show_most_informative_features(vectorizer2.get_feature_names_out(), bigram_model, test_data.values[:, 1],test_data.values[:, 0])

predictions = (bigram_model.predict(test_bigram))
print(predictions)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))