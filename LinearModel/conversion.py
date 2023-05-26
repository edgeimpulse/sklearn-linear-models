import io, os, shutil
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

def convert_jax(input_shape, jax_fn, out_file):
    full_shape = tuple([ 1 ]) + tuple(input_shape)
    reference_input = jnp.zeros(full_shape)

    converter = tf.lite.TFLiteConverter.experimental_from_jax(
        [jax_fn], [[('input1', reference_input)]])

    tflite_model = converter.convert()
    with open(out_file, 'wb') as f:
        f.write(tflite_model)
