import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.linear_model import Ridge
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


print('Training model on', str(X_train.shape[0]), 'inputs...')
# train your model
reg = Ridge(alpha=1.0, random_state=RANDOM_SEED)
reg.fit(X_train, Y_train)
print('Training model OK')
print('')

print('Mean accuracy (training set):', reg.score(X_train, Y_train))
print('Mean accuracy (validation set):', reg.score(X_test, Y_test))
print('')

# here comes the magic, provide a JAX version of the `proba` function
def minimal_predict(X):
    # first the linear model
    # see: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_base.py#L430
    out = jnp.dot(X, reg.coef_.T) + reg.intercept_
    return out.reshape((-1, 1)) 


print('Converting model...')
convert_jax(X_train.shape[1:], minimal_predict, os.path.join(args.out_directory, 'model.tflite'))
print('Converting model OK')
print('')
