import sklearn  # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import pickle

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("script arguments:")
print(sys.argv)

# Load files
parser = argparse.ArgumentParser(description="Train custom ML model")
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--out-directory", type=str, required=True)
parser.add_argument("--alphas", type=str, required=True)

args, _ = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, "X_split_train.npy"))
Y_train = np.load(os.path.join(args.data_directory, "Y_split_train.npy"))
X_test = np.load(os.path.join(args.data_directory, "X_split_test.npy"))
Y_test = np.load(os.path.join(args.data_directory, "Y_split_test.npy"))


# # sparse representation of the labels (1-based)
Y_train = np.argmax(Y_train, axis=1) + 1
Y_test = np.argmax(Y_test, axis=1) + 1

print("Training model on", str(X_train.shape[0]), "inputs...")
# train your model
clf = RidgeClassifierCV(alphas=eval(args.alphas))
clf.fit(X_train, Y_train)
print("Training model OK")
print("")

print("Mean accuracy (training set):", clf.score(X_train, Y_train))
print("Mean accuracy (validation set):", clf.score(X_test, Y_test))
print("")


# here comes the magic, provide a JAX version of the `proba` function
def minimal_predict_proba(X):
    # sklearn only outputs 1d array of coefficients for binary classification but we need coefficients for each label to be compatible with down stream tasks
    if clf.coef_.shape[0] == 1:
        clf.coef_ = jnp.vstack([clf.coef_ * -1, clf.coef_])
        clf.intercept_ = jnp.hstack([clf.intercept_ * -1, clf.intercept_])
    # first the linear model
    # see: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_base.py#L430
    y = jnp.dot(X, clf.coef_.T) + clf.intercept_

    # then the exponentiation
    # see https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_base.py#L462
    # and https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/special/_logit.h#L15
    expit = jnp.exp(y)

    # and finally the normalisation
    # see: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_base.py#L467
    prob = expit / expit.sum(axis=1, keepdims=True)
    return prob

print("Saving pickle model...")
with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(clf,f)
print("Saving model OK")
print("")