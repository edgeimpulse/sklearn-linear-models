import sklearn  # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from conversion import convert_jax
import jax.numpy as jnp
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
parser.add_argument("--solver", type=str, required=True)
parser.add_argument("--lbfgs-penalty", type=str)
parser.add_argument("--liblinear-penalty", type=str)
parser.add_argument("--newton-cg-penalty", type=str)
parser.add_argument("--newton-cholesky-penalty", type=str)
parser.add_argument("--sag-penalty", type=str)
parser.add_argument("--saga-penalty", type=str)
parser.add_argument("--l1-ratio", type=str)

args, _ = parser.parse_known_args()

if args.solver == "lbfgs":
    penalty = args.lbfgs_penalty
elif args.solver == "liblinear":
    penalty = args.liblinear_penalty
elif args.solver == "newton-cg":
    penalty = args.newton_cg_penalty
elif args.solver == "newton-cholesky":
    penalty = args.newton_cholesky_penalty
elif args.solver == "sag":
    penalty = args.sag_penalty
elif args.solver == "saga":
    penalty = args.saga_penalty

if penalty == "None":
    penalty = None

if penalty == "elasticnet":
    l1_ratio = float(args.l1_ratio)
else:
    l1_ratio = None


print(
    f"Training with {args.solver} solver and {penalty} penalty and {l1_ratio} l1 ratio"
)
out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, "X_split_train.npy"))
Y_train = np.load(os.path.join(args.data_directory, "Y_split_train.npy"))
X_test = np.load(os.path.join(args.data_directory, "X_split_test.npy"))
Y_test = np.load(os.path.join(args.data_directory, "Y_split_test.npy"))

# sparse representation of the labels (1-based)
Y_train = np.argmax(Y_train, axis=1) + 1
Y_test = np.argmax(Y_test, axis=1) + 1

print("Training model on", str(X_train.shape[0]), "inputs...")
# train your model
clf = LogisticRegression(
    penalty=penalty, solver=args.solver, random_state=RANDOM_SEED, l1_ratio=l1_ratio
).fit(X_train, Y_train)
print("Training model OK")
print("")

print("Mean accuracy (training set):", clf.score(X_train, Y_train))
print("Mean accuracy (validation set):", clf.score(X_test, Y_test))
print("")


# here comes the magic, provide a JAX version of the `proba` function
def minimal_predict_proba(X):
    # create extra dimension when using 2 class labels to fix downstream tflite profiling/model testing
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
    pickle.dump(clf, f)
print("Saving model OK")
print("")