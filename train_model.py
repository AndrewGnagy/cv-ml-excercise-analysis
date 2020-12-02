import numpy as np
import tensorflow as tf
import pickle

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
    return tf.keras.Sequential([
        tf.keras.Input(shape=(max_len,), dtype='float32', name='coords', ragged=True),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
#   inputs = tf.keras.Input(
#       shape=[None], dtype='float64', name='coords', ragged='true')
#   in_flat = tf.keras.layers.Flatten()(inputs)
#   dense_layer = tf.keras.layers.Dense(20, activation='relu')(in_flat)
#   outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
#   return tf.keras.Model(inputs, outputs)


data = pickle.load(open("datafile-flat.data", 'rb'))

ragged_tensor = tf.ragged.constant(np.asarray(data['train']))
padded_set = ragged_tensor.to_tensor()
model = make_model(data['max_len'])
model.summary()
model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]
history = model.fit(
    padded_set,
    np.asarray(data['labels']).astype('float32'),
    epochs=20,
    verbose=1,
    shuffle=True)
results = model.predict(np.asarray(padded_set))
print(results)
model.save("models/squat_model0")


