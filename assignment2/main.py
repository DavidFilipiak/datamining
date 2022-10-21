import os
import string
import pandas as pd
import re
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(folds):

    """ Extracts the reviews and their associated labels from the op_spam dataset from specified
    fold folders in the repository.
    Assigns the label 0 to deceptive reviews and label 1 to truthful reviews.

    Args:
        folds: list of folds in data set from which to extract reviews
    Returns:
        labels: list of labels
        reviews: list of raw review text
    """
    labels = []
    reviews = []
    for root,dirs,files in os.walk('op_spam_v1.4/negative_polarity/'):
        if root[-5:] in folds:
            for file in files:
                if file.startswith('t'):
                    labels.append(1)
                else:
                    labels.append(0)
                f = open(os.path.join(root,file), 'r')
                reviews.append(f.read())
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
        #r = [i for i in tokens if not i in ENGLISH_STOP_WORDS]
        #r = [lemmatizer.lemmatize(word) for word in r]
        preprocessed_reviews.append(r)

    return preprocessed_reviews

def load_data_in_frame(folds):
    """Creates dataframe with labels and reviews.

    Args:
        folds: list of folds in data set from which to extract reviews
    Returns:
        df: dataframe with labels and reviews
    """
    labels, reviews = load_data(folds)
    print(reviews[0])

    preprocessed_reviews = preprocess_data(reviews)
    label_df = pd.DataFrame(labels, columns=['label'])
    #review_df = pd.DataFrame(reviews, columns=['review'])
    pre_review_df = pd.DataFrame(preprocessed_reviews, columns=['preprocessed review'])
    df = pd.merge(label_df, pre_review_df, right_index=True, left_index=True)
    return df

train_folds = ['fold1','fold2','fold3','fold4']
test_folds = ['fold5']

train_data = load_data_in_frame(train_folds)
test_data = load_data_in_frame(test_folds)

# to check:
print(len(train_data),len(test_data))
print(train_data.head())
print(test_data.head())


