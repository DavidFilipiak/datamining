import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

def forest_analysis(X_train, y_train):

    model = RandomForestClassifier(random_state=42)
    parameters = {'n_estimators': [200,400,600,800],
                  #'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
                  'criterion': ['gini', 'entropy']}

    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
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
vectorizer = CountVectorizer(analyzer="word", stop_words = "english")
train_unigram = vectorizer.fit_transform(train_data.values[:, 1])
test_unigram = vectorizer.transform(test_data.values[:, 1])

unigram_model = forest_analysis(train_unigram, list(train_data.values[:, 0]))

print("\nUnigram Model Analysis: \n")
predictions = (unigram_model.predict(test_unigram))
print(predictions)
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

# Bigram
vectorizer2 = CountVectorizer(analyzer="word", stop_words = "english", ngram_range=(2, 2))
train_bigram = vectorizer2.fit_transform(train_data.values[:, 1])
test_bigram = vectorizer2.transform(test_data.values[:, 1])

bigram_model = forest_analysis(train_bigram, list(train_data.values[:, 0]))

print("\nBigram Model Analysis: \n")
predictions = (bigram_model.predict(test_bigram))
print(predictions)
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))