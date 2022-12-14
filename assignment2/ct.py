import main
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

def tree_analysis(X_train, y_train):

    model = tree.DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'],
                  'ccp_alpha': [0.001, 0.005, 0.01, 0.05, 0.1]}

    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
    grid.fit(X_train,y_train)

    print('Best parameters: \n', grid.best_params_)
    print('Best score: \n', grid.best_score_)

    best_model = grid.best_estimator_

    return best_model

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

unigram_model = tree_analysis(train_unigram, list(train_data.values[:, 0]))


predictions = (unigram_model.predict(test_unigram))
print('predictions:', predictions)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
main.add_prediction_for_mcnemar(predictions, "ct", "unigram")

# Bigram
print("\nUnigram + Bigram Model Analysis: \n")
vectorizer2 = CountVectorizer(analyzer="word", stop_words = "english", ngram_range=(1, 2))
train_bigram = vectorizer2.fit_transform(train_data.values[:, 1])
test_bigram = vectorizer2.transform(test_data.values[:, 1])

bigram_model = tree_analysis(train_bigram, list(train_data.values[:, 0]))


predictions_2 = (bigram_model.predict(test_bigram))
print('predictions:', predictions_2)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions_2)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions_2)))
main.add_prediction_for_mcnemar(predictions_2, "ct", "bigram")





