import numpy as np
import tensorflow as tf
import pickle



def give_prediction(datafile):
    data = pickle.load(open(datafile, 'rb'))
    #Normalize vector length (Will cut off data!)
    #TODO remove magic numbers or adapt ragged input to model
    data_len = len(data['train'][0])
    target_len = 2482
    if data_len > target_len:
        data['train'][0] = data['train'][0][:target_len]
    if data_len < target_len:
        data['train'][0] = np.pad(data['train'][0], (0, target_len - data_len), 'constant', constant_values=(0, 0))
    ragged_tensor = tf.ragged.constant(np.asarray(data['train']))
    padded_set = ragged_tensor.to_tensor()

    model = tf.keras.models.load_model("models/squat_model0")
    results = model.predict(np.asarray(padded_set))
    print(results[0][0])
    prediction = results[0][0] > 0.5 and results[0][0] < 1.75
    return "good" if prediction else "bad"

#give_prediction("data/jjacks-flat.data")