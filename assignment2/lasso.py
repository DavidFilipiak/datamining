import main
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso

#Data preparation
def remove_stopping_words(data):
    return [word for word in data.split()]
    #return [word for word in data.split() if word not in ENGLISH_STOP_WORDS]

train_folds = ['fold1', 'fold2', 'fold3', 'fold4']
test_folds = ['fold5']

train_data = main.load_data_in_frame(train_folds)
test_data = main.load_data_in_frame(test_folds)

vectorizer = CountVectorizer(analyzer=remove_stopping_words)
train_transformed = vectorizer.fit_transform(train_data.values[:, 1])
test_transformed = vectorizer.transform(test_data.values[:, 1])

X_train = train_transformed
y_train = list(train_data.values[:, 0])
X_test = test_transformed
y_test = list(test_data.values[:, 0])

## Unigram Model creation

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)
# Fit model
model.fit(X_train, y_train)

# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

plt.semilogx(model.alphas_, model.mse_path_, ":")
print(len(model.alphas_))
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

plt.show()
