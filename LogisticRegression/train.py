import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from conversion import convert_jax
import jax.numpy as jnp

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Train custom ML model')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, _ = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'))
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'))
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

# sparse representation of the labels (1-based)
Y_train = np.argmax(Y_train, axis=1) + 1
Y_test = np.argmax(Y_test, axis=1) + 1

print('Training model on', str(X_train.shape[0]), 'inputs...')
# train your model
clf = LogisticRegression(random_state=RANDOM_SEED).fit(X_train, Y_train)
print('Training model OK')
print('')

print('Mean accuracy (training set):', clf.score(X_train, Y_train))
print('Mean accuracy (validation set):', clf.score(X_test, Y_test))
print('')

# here comes the magic, provide a JAX version of the `proba` function
def minimal_predict_proba(X):
    # create extra dimension when using 2 class labels to fix downstream tflite profiling/model testing
    if clf.coef_.shape[0] == 1:
        clf.coef_ = jnp.vstack([clf.coef_ * -1,clf.coef_])
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

print('Converting model...')
convert_jax(X_train.shape[1:], minimal_predict_proba, os.path.join(args.out_directory, 'model.tflite'))
print('Converting model OK')
print('')
