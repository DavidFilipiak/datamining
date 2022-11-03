import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

def forest_analysis(X_train, y_train):

    model = RandomForestClassifier(random_state=42)
    parameters = {'n_estimators': [200,400,600,800],
                  'ccp_alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
                  'criterion': ['gini', 'entropy']}

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

unigram_model = forest_analysis(train_unigram, list(train_data.values[:, 0]))


predictions = (unigram_model.predict(test_unigram))
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

# Bigram
print("\nUnigram + Bigram Model Analysis: \n")
vectorizer2 = CountVectorizer(analyzer="word", stop_words = "english", ngram_range=(1, 2))
train_bigram = vectorizer2.fit_transform(train_data.values[:, 1])
test_bigram = vectorizer2.transform(test_data.values[:, 1])

bigram_model = forest_analysis(train_bigram, list(train_data.values[:, 0]))


predictions_2 = (bigram_model.predict(test_bigram))
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions_2)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions_2)))