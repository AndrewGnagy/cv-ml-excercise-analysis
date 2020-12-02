from itertools import repeat
import numpy as np
import tensorflow as tf
import pickle
import random

def pad_sequence(sequence, max_seq_length):
    """Pads the input sequence (a `tf.SparseTensor`) to `max_seq_length`."""
    pad_size = tf.maximum([0], max_seq_length - tf.shape(sequence)[0])
    padded = tf.concat(
        [sequence.values,
            tf.fill((pad_size), tf.cast(0, sequence.dtype))],
        axis=0)
    # The input sequence may be larger than max_seq_length. Truncate down if
    # necessary.
    return tf.slice(padded, [0], [max_seq_length])


def make_model(max_len):
    conv_2d_layer = tf.keras.layers.Conv2D(64, (17, 2))
    return tf.keras.Sequential([
        tf.keras.Input(shape=(None, 17, 2), dtype='float32', name='coords', ragged=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(17, activation='relu')),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
#   inputs = tf.keras.Input(
#       shape=[None], dtype='float64', name='coords', ragged='true')
#   in_flat = tf.keras.layers.Flatten()(inputs)
#   dense_layer = tf.keras.layers.Dense(20, activation='relu')(in_flat)
#   outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
#   return tf.keras.Model(inputs, outputs)


data = pickle.load(open("datafile.data", 'rb'))

#Fill with -1s
# padded_set = []
# for dataset in data['train']:
#     padded_set.append(np.pad(dataset, (0, data['max_len'] - len(dataset)), 'constant', constant_values=(-1.0, -1.0)))
# print(padded_set)
#ragged_tensor = tf.RaggedTensor.from_tensor(np.asarray(padded_set), padding=-1.0, row_splits_dtype=tf.dtypes.float32)
ragged_tensor = tf.ragged.constant(np.asarray(data['train']))
model = make_model(data['max_len'])
model.summary()
model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]
history = model.fit(
    ragged_tensor,
    np.asarray(data['labels']).astype('float32'),
    epochs=10,
    verbose=1,
    shuffle=True)
results = model.predict(np.asarray([padded_set[3]]))
print(results)


