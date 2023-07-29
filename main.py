import os
import ast
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
import music21 as m21
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras import backend as k
from keras.optimizers import Adam
from keras.src.utils import plot_model
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.metrics import SparseCategoricalAccuracy
from sklearn.model_selection import train_test_split
from keras.losses import SparseCategoricalCrossentropy


tf.get_logger().setLevel(logging.ERROR)
k.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")  # ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_duration_model(dataset="Soprano", epochs=100):
    """Trains the note duration model for the specified dataset"""
    df = pd.read_csv(f"Data\\Tabular\\{dataset}.csv", sep=';')
    df = df[['event', 'time']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df = df[df['time'].apply(len) > 0]  # Remove empty sequences
    df = df[df['event'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())

    # The model will take in the event and the previous time as input and predict the next time
    df['time_prev'] = df['time'].apply(lambda x: np.concatenate([[0], x[:-1]]))
    df['time_next'] = df['time'].apply(lambda x: np.concatenate([x[1:], [0]]))

    inputs = np.array(df[['event', 'time_prev']])
    outputs = np.array(df['time_next'])

    # Find the longest event array (index 0) and longest time array (index 1)
    # Pad all other arrays to this length using -1 (rest) for event and 0.0 for time
    max_event_len = max([len(x[0]) for x in inputs])
    max_time_len = max([len(x[1]) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len - len(x), -1)]).astype(int)
                         for x in np.array(df['event'])])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len - len(x), 0.)]).astype(float)
                         for x in np.array(df['time_prev'])])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.array([np.concatenate([x, np.full(max_output_len - len(x), 0.)]).astype(float) for x in outputs])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Bi-Directonal LSTM
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(max_event_len, 2)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    model.summary()
    plot_model(model, to_file=f'Images\\{dataset}_duration_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Save the model, scaler, and max length (all 3 max lengths should be the same)
    model.save(f"Weights\\Duration\\{dataset}.h5")
    pkl.dump(scaler, open(f"Weights\\Duration\\{dataset}_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights\\Duration\\{dataset}_seq_len.pkl", 'wb'))

    # Plot the training history
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title(f"{dataset} Duration Model Loss/MSE")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(model.history.history['mae'])
    plt.plot(model.history.history['val_mae'])
    plt.title(f"{dataset} Duration Model MAE")
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Test the model with the first event array; pad the arrays to match the max lengths and then trim after prediction
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time_prev = np.concatenate([df['time_prev'][0], np.full(max_time_len - len(df['time_prev'][0]), 0.)]).astype(float)
    time_next = df['time_next'][0]
    # Reshape the input to match the model input layer
    input_data = np.array([event, time_prev]).reshape(1, max_event_len, 2)
    time_pred = model.predict(input_data)
    # Rescale the time to match the original data
    time_next = scaler.inverse_transform(time_next.reshape(-1, 1)).flatten()
    time_pred = time_pred.squeeze()  # Remove dimensions of size 1
    time_pred = time_pred[:len(time_next)]  # Trim the predicted time to match the actual time
    time_pred = scaler.inverse_transform(time_pred.reshape(-1, 1)).flatten()
    print("Actual time: ", time_next)
    print("Predicted time: ", time_pred)
    print("Difference: ", time_next - time_pred)

    return model, scaler, max_event_len


if __name__ == '__main__':
    print("Hello world!")
    voices = ["Soprano", "Alto", "Tenor", "Bass"]
    for dataset in voices:
        train_duration_model(dataset)
    # train()
