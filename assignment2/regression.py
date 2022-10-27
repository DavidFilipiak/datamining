import main
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import GridSearchCV  

def RegressionAnalysis(X_train, y_train):
    # Basic model without any tuned parameters
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter= 100)
    # Fit model
    model.fit(X_train, y_train)
    parameters = {'C': [x / 10 for x in range(0, 100)],              
    }
    #Create grid serach to find the best C value
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=3, refit=True) 
    grid.fit(X_train, y_train)

    scores_df = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_

    print("Best params: \n", best_params, "\nBest score: ", grid.best_score_)

    best_model = grid.best_estimator_

    #Plot
    plt.semilogx(scores_df['param_C'], scores_df['mean_test_score'], ":")
    plt.plot(
        scores_df['param_C'] ,
        scores_df['mean_test_score'],
        "k",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(
        best_params['C'], linestyle="--", color="k", label="C: best estimate"
    )

    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Mean test score")
    plt.title("Mean test score")
    plt.axis("tight")

    plt.show()
    return best_model

## Data preparation
train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)


## Unigram Model
vectorizer = CountVectorizer(analyzer="word", stop_words = "english")
train_unigram = vectorizer.fit_transform(train_data.values[:, 1])
test_unigram = vectorizer.transform(test_data.values[:, 1])

unigram_model = RegressionAnalysis(train_unigram, list(train_data.values[:, 0]))

print("\nUnigram Model Analysis: \n")
predictions = (unigram_model.predict(test_unigram))
print(predictions)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))

# Bigram Model
vectorizer2 = CountVectorizer(analyzer="word", stop_words = "english", ngram_range=(2, 2))
train_bigram = vectorizer2.fit_transform(train_data.values[:, 1])
test_bigram = vectorizer2.transform(test_data.values[:, 1])

print(vectorizer.get_feature_names_out())
print(vectorizer2.get_feature_names_out())
bigram_model = RegressionAnalysis(train_bigram, list(train_data.values[:, 0]))

print("\nBigram Model Analysis: \n")
predictions = bigram_model.predict(test_bigram)
print("Classification report: \n", classification_report(list(test_data.values[:, 0]), list(predictions)))
print("Confusion Matrix: \n", confusion_matrix(list(test_data.values[:, 0]), list(predictions)))
