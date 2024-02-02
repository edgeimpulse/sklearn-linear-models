import sklearn  # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.linear_model import RidgeClassifier
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
parser.add_argument("--alpha", type=float, required=True)

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
clf = RidgeClassifier(alpha=args.alpha, random_state=RANDOM_SEED)
clf.fit(X_train, Y_train)
print("Training model OK")
print("")

print("Mean accuracy (training set):", clf.score(X_train, Y_train))
print("Mean accuracy (validation set):", clf.score(X_test, Y_test))
print("")

print("Saving model...")
with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(clf,f)
print("Saving model OK")
print("")
