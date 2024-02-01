import sklearn  # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging, ast
import numpy as np
from sklearn.linear_model import LinearRegression
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
parser.add_argument("--fit-intercept", type=bool, required=True)

args, _ = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, "X_split_train.npy"))
Y_train = np.load(os.path.join(args.data_directory, "Y_split_train.npy"))
X_test = np.load(os.path.join(args.data_directory, "X_split_test.npy"))
Y_test = np.load(os.path.join(args.data_directory, "Y_split_test.npy"))


print("Training model on", str(X_train.shape[0]), "inputs...")
# train your model
reg = LinearRegression(fit_intercept=args.fit_intercept)
reg.fit(X_train, Y_train)
print("Training model OK")
print("")

print("Mean accuracy (training set):", reg.score(X_train, Y_train))
print("Mean accuracy (validation set):", reg.score(X_test, Y_test))
print("")


# here comes the magic, provide a JAX version of the `proba` function
def minimal_predict(X):
    # first the linear model
    # see: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_base.py#L430
    out = jnp.dot(X, reg.coef_.T) + reg.intercept_
    # reshape required for downstream tflite model
    return out.reshape((-1, 1))

print("Saving pickle model...")
with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(reg,f)
print("Saving model OK")
print("")
