import os
import pandas as pd

def load_data(path,folds):
    labels = []
    reviews = []
    for root,dirs,files in os.walk(path):
        if root[-5:] in folds:
            for file in files:
                if file.startswith('t'):
                    labels.append(1)
                else:
                    labels.append(0)
                f = open(os.path.join(root,file), 'r')
                reviews.append(f.read())
    return labels, reviews

def load_data_in_frame(path, folds):
    labels, reviews = load_data(path, folds)
    label_df = pd.DataFrame(labels, columns=['label'])
    review_df = pd.DataFrame(reviews, columns=['review'])
    df = pd.merge(label_df, review_df, right_index=True, left_index=True)
    return df

path = 'op_spam_v1.4/negative_polarity/'
train_folds = ['fold1','fold2','fold3','fold4']
test_folds = ['fold5']

train_data = load_data_in_frame(path, train_folds)
test_data = load_data_in_frame(path, test_folds)

# to check:
print(len(train_data),len(test_data))

