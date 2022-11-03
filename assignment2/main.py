import os
import string
import pandas as pd
import re
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(folds, scope=2):

    """ Extracts the reviews and their associated labels from the op_spam dataset from specified
    fold folders in the repository.
    Assigns the label 0 to deceptive reviews and label 1 to truthful reviews.

    Args:
        folds: list of folds in data set from which to extract reviews
        scope: 0 for only deceptive reviews, 1 for only truthful reviews, 2 for all reviews
    Returns:
        labels: list of labels
        reviews: list of raw review text
    """
    labels = []
    reviews = []
    for root,dirs,files in os.walk('op_spam_v1.4/negative_polarity/'):
        if root[-5:] in folds:
            for file in files:
                if scope == 2:
                    if file.startswith('t'):
                        labels.append(1)
                    else:
                        labels.append(0)
                    f = open(os.path.join(root,file), 'r')
                    reviews.append(f.read())
                    f.close()
                elif scope == 1:
                    if file.startswith('t'):
                        labels.append(1)
                        f = open(os.path.join(root, file), 'r')
                        reviews.append(f.read())
                        f.close()
                elif scope == 0:
                    if file.startswith('d'):
                        labels.append(0)
                        f = open(os.path.join(root, file), 'r')
                        reviews.append(f.read())
                        f.close()
    return labels, reviews

def preprocess_data(reviews):
    preprocessed_reviews = []

    #lemmatizer = WordNetLemmatizer()

    for r in reviews:
        r = r.lower()
        r = re.sub(r'\d+', '', r)
        r = r.translate(str.maketrans('', '', string.punctuation))
        r = r.strip()
        #tokens = word_tokenize(r)
        #r = [i for i in r if i not in ENGLISH_STOP_WORDS]
        #r = [lemmatizer.lemmatize(word) for word in r]
        preprocessed_reviews.append(r)

    return preprocessed_reviews

def load_data_in_frame(folds, scope=2):
    """Creates dataframe with labels and reviews.

    Args:
        folds: list of folds in data set from which to extract reviews
        scope: 0 for only deceptive reviews, 1 for only truthful reviews, 2 for all reviews
    Returns:
        df: dataframe with labels and reviews
    """
    labels, reviews = load_data(folds, scope)
    print(reviews[0])

    preprocessed_reviews = preprocess_data(reviews)
    label_df = pd.DataFrame(labels, columns=['label'])
    #review_df = pd.DataFrame(reviews, columns=['review'])
    pre_review_df = pd.DataFrame(preprocessed_reviews, columns=['preprocessed review'])
    df = pd.merge(label_df, pre_review_df, right_index=True, left_index=True)
    return df

def calculate_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    ##Acuracy
    acc = (confusion_matrix[1,1] + confusion_matrix[0,0]) / confusion_matrix.sum()
    print("Accuracy: ", acc)

    ## Precision
    prc = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
    print("Precision: ", prc)

    ## Recall
    rcall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
    print("Recall: ", rcall)

    ## F1 Score
    f1 = 2 * (prc * rcall) / (prc + rcall)
    print("F1 score: ", f1)

# pefrom McNemar statistical test on the provided models
# predictions_a -> first model for testing
# predictions_b -> second model for testing
# classification -> the labels to compare the models to
def performMcNemar(predictions_a, predictions_b, classification):
    i = 0
    no_no = 0
    no_yes = 0
    yes_no = 0
    yes_yes = 0

    while i < len(predictions_a):
        if classification[i] == predictions_a[i] and predictions_a[i] == predictions_b[i]:
            yes_yes += 1
        if classification[i] != predictions_a[i] and classification[i] != predictions_b[i]:
            no_no += 1
        if classification[i] == predictions_a[i] and classification[i] != predictions_b[i]:
            yes_no += 1
        if classification[i] == predictions_b[i] and classification[i] != predictions_a[i]:
            no_yes += 1
        i += 1
    ## define contingency table
    table = [[yes_yes, yes_no],
    		 [no_yes, no_no]]
    print(table)
    result = mcnemar(table, exact=True)
    print('statistic=%.5f, p-value=%.35f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

# model = accepted values: mnb, regression, ct, forest
# type = values: unigram, bigram
def add_prediction_for_mcnemar(predictions, model, type):
    file = ""
    if type == "unigram":
        file = "unigram_predictions.txt"
    else:
        file = "uni_bigram_predictions.txt"

    lines = []
    with open(file, "r", encoding="utf-8") as read:
        lines = read.readlines()

    with open(file, "w", encoding="utf-8") as write:
        for line in lines:
            if line.startswith(model):
                new_line = str(model) + ":" + "".join([str(x) for x in list(predictions)])
                write.write(new_line + "\n")
            else:
                write.write(line)



if __name__ == "__main__":
    train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
    test_folds = ['fold5']

    train_data = load_data_in_frame(train_folds)
    test_data = load_data_in_frame(test_folds)

    # to check:
    print(len(train_data), len(test_data))
    print(train_data.head())
    print(test_data.head())

