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
from Performer import *
from keras import layers
from keras import backend as k
from keras.optimizers import Adam
from keras.src.utils import plot_model
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from data_utils import key_signature_to_number
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


def plot_histories(model, feature1, feature2, title, ylabel, filename=None):
    plt.plot(model.history.history[feature1])
    plt.plot(model.history.history[feature2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if filename is not None:
        plt.savefig(filename)


def train_composition_model(dataset="Soprano", epochs=100):
    """Trains a Transformer model to generate notes and times for a given key, tempo, and time signature."""
    df = pd.read_csv(f"Data\\Tabular\\{dataset}.csv", sep=';')
    df = df[['event', 'time', 'tempo', 'time_signature_count', 'time_signature_beat', 'key_signature']]
    # event;velocity;time;tempo;time_signature_count;time_signature_beat;key_signature

    # region Preprocessing
    if dataset in ["Alto", "Tenor"]:
        df_S = pd.read_csv(f"Data\\Tabular\\Soprano.csv", sep=';')
        df_B = pd.read_csv(f"Data\\Tabular\\Bass.csv", sep=';')
        df_S = df_S[['event', 'time']]
        df_B = df_B[['event', 'time']]
        # Concatenate to main dataframe; rename columns to include voice part
        df_S.columns = [f"{x}_S" for x in df_S.columns]
        df_B.columns = [f"{x}_B" for x in df_B.columns]
        df = pd.concat([df, df_S, df_B], axis=1)

    # Normalize the data
    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval).apply(np.array)
        df = df[df[col].apply(len) > 0]
    for col in ['time_signature_count', 'time_signature_beat', 'key_signature']:
        df[col] = df[col].apply(lambda x: np.array([int(y) for y in x]))
    # Load scalers
    with open("Weights\\Tempo\\tempo_scaler.pkl", "rb") as f:
        tempo_scaler = pkl.load(f)
    with open("Weights\\TimeSignature\\time_sig_scaler.pkl", "rb") as f:
        time_sig_scaler = pkl.load(f)
    with open("Weights\\KeySignature\\key_scaler.pkl", "rb") as f:
        key_sig_scaler = pkl.load(f)
    with open(f"Weights\\Duration\\{dataset}_time_scaler.pkl", "rb") as f:
        time_scaler = pkl.load(f)
    sop_scaler, bass_scaler = None, None
    if dataset in ["Alto", "Tenor"]:
        with open("Weights\\Duration\\Soprano_time_scaler.pkl", "rb") as f:
            sop_scaler = pkl.load(f)
        with open("Weights\\Duration\\Bass_time_scaler.pkl", "rb") as f:
            bass_scaler = pkl.load(f)
    # Apply scalers
    df['time'] = df['time'].apply(lambda x: time_scaler.transform(x.reshape(-1, 1)).flatten())
    df['tempo'] = df['tempo'].apply(lambda x: tempo_scaler.transform(x.reshape(-1, 1)).flatten())
    df['time_signature_count'] = \
        df['time_signature_count'].apply(lambda x: time_sig_scaler.transform(x.reshape(-1, 1)).flatten())
    df['time_signature_beat'] = \
        df['time_signature_beat'].apply(lambda x: time_sig_scaler.transform(x.reshape(-1, 1)).flatten())
    df['key_signature'] = df['key_signature'].apply(lambda x: key_sig_scaler.transform(x.reshape(-1, 1)).flatten())
    if dataset in ["Alto", "Tenor"] and (sop_scaler is not None and bass_scaler is not None):
        df['time_S'] = df['time_S'].apply(lambda x: sop_scaler.transform(x.reshape(-1, 1)).flatten())
        df['time_B'] = df['time_B'].apply(lambda x: bass_scaler.transform(x.reshape(-1, 1)).flatten())

    # Redundant for this function -- just for reference
    # inputs = df[['tempo', 'time_signature_count', 'time_signature_beat', 'key_signature']]
    # outputs = np.array(df[['event', 'time']])
    # if dataset in ["Alto", "Tenor"]:
    #     inputs = pd.concat([inputs, df[['event_S', 'event_B', 'time_S', 'time_B']]], axis=1)
    # inputs = np.array(inputs)

    # Grab the maximum sequence length and pad the rest; they should all be the same length by now
    with open(f"Weights\\Duration\\{dataset}_seq_len.pkl", "rb") as f:
        max_seq_len = pkl.load(f)
    inputs_tempo = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0.)]).astype(float)
                             for x in np.array(df['tempo'])])
    inputs_time_n = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0)]).astype(int)
                              for x in np.array(df['time_signature_count'])])
    inputs_time_d = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0)]).astype(int)
                              for x in np.array(df['time_signature_beat'])])
    inputs_key = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0)]).astype(int)
                           for x in np.array(df['key_signature'])])
    if dataset in ["Alto", "Tenor"]:
        inputs_e_S = np.array([np.concatenate([x, np.full(max_seq_len-len(x), -1)]).astype(int)
                               for x in np.array(df['event_S'])])
        inputs_t_S = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0.)]).astype(float)
                               for x in np.array(df['time_S'])])
        inputs_e_B = np.array([np.concatenate([x, np.full(max_seq_len-len(x), -1)]).astype(int)
                               for x in np.array(df['event_B'])])
        inputs_t_B = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0.)]).astype(float)
                               for x in np.array(df['time_B'])])
        inputs = np.stack((inputs_tempo, inputs_time_n, inputs_time_d, inputs_key,
                           inputs_e_S, inputs_t_S, inputs_e_B, inputs_t_B), axis=-1)
    else:
        inputs = np.stack((inputs_tempo, inputs_time_n, inputs_time_d, inputs_key), axis=-1)
    outputs_e = np.array([np.concatenate([x, np.full(max_seq_len-len(x), -1)]).astype(int)
                          for x in np.array(df['event'])])
    outputs_t = np.array([np.concatenate([x, np.full(max_seq_len-len(x), 0.)]).astype(float)
                          for x in np.array(df['time'])])
    outputs = np.stack((outputs_e, outputs_t), axis=-1)
    # endregion Preprocessing

    group_size = 25
    inputs = inputs.reshape((-1, group_size, inputs.shape[-1]))
    outputs = outputs.reshape((-1, group_size, outputs.shape[-1]))

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Build an array of each feature from X and y to use as input for the model
    y_train_e = y_train[:, :, 0]
    y_train_t = y_train[:, :, 1]
    y_test_e = y_test[:, :, 0]
    y_test_t = y_test[:, :, 1]
    # X_train_split = [X_train[:, :, i] for i in range(X_train.shape[-1])]
    # X_test_split = [X_test[:, :, i] for i in range(X_test.shape[-1])]

    # y_train_e = y_train[:, :, 0, np.newaxis]
    # y_train_t = y_train[:, :, 1, np.newaxis]
    # y_test_e = y_test[:, :, 0, np.newaxis]
    # y_test_t = y_test[:, :, 1, np.newaxis]
    X_train_split = [X_train[:, :, i, np.newaxis] for i in range(X_train.shape[-1])]
    X_test_split = [X_test[:, :, i, np.newaxis] for i in range(X_test.shape[-1])]

    # Print the shapes of all datasets
    # print(f"y_train_e shape: {y_train_e.shape}")
    # print(f"y_test_e shape: {y_test_e.shape}")
    # print(f"y_train_t shape: {y_train_t.shape}")
    # print(f"y_test_t shape: {y_test_t.shape}")
    # print(f"X_train_split shape: {[x.shape for x in X_train_split]}")
    # print(f"X_test_split shape: {[x.shape for x in X_test_split]}")

    # Transformer model with two output layers (one for event, one for time)
    model = Performer(
        num_layers=2,
        d_model=group_size,  # 512
        num_heads=8,
        dff=2048,
        features=len(X_train_split),
        input_vocab_size=10000,
        target_vocab_size=10000,
        rate=0.1
    )
    checkpoint_path = f"Weights\\Composition\\{dataset}_checkpoint.ckpt"
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', 'mse'], metrics=['accuracy', 'mse'])
    model.build(input_shape=(None, group_size, len(X_train_split), 1))
    model.summary()
    plot_model(model, to_file=f'Images\\{dataset}_composition_model.png',
               show_shapes=True, show_layer_names=True, expand_nested=True)

    model.fit(X_train_split, [y_train_e, y_train_t], callbacks=[callback, early_stop],
              validation_data=(X_test_split, [y_test_e, y_test_t]), epochs=epochs, batch_size=32)
    plot_histories(model, 'loss', 'val_loss', f"{dataset} Composition Model Loss (SCC)", 'Loss (SCC)')
    plot_histories(model, 'accuracy', 'val_accuracy', f"{dataset} Composition Model Accuracy", 'Accuracy')

    model.save_weights(f"Weights\\Composition\\{dataset}_model.png.h5")

    pass


def train_duration_model(dataset="Soprano", epochs=100):
    """Trains a Bi-LSTM model to predict the duration of the next note given the previous note and duration."""
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
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int)
                         for x in np.array(df['event'])])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float)
                         for x in np.array(df['time_prev'])])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0.)]).astype(float) for x in outputs])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Bi-Directonal LSTM
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(max_event_len, 2)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images\\{dataset}_duration_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', f"{dataset} Duration Model Loss (MSE)", 'Loss (MSE)')
    plot_histories(model, 'mae', 'val_mae', f"{dataset} Duration Model MAE", 'MAE')

    # Save the model, scaler, and max length (all 3 max lengths should be the same)
    model.save(f"Weights\\Duration\\{dataset}_model.h5")
    pkl.dump(scaler, open(f"Weights\\Duration\\{dataset}_time_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights\\Duration\\{dataset}_seq_len.pkl", 'wb'))

    # Test the model with the first event array; pad the arrays to match the max lengths and then trim after prediction
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time_prev = np.concatenate([df['time_prev'][0], np.full(max_time_len - len(df['time_prev'][0]), 0.)]).astype(float)
    time_next = df['time_next'][0]
    # Reshape the input to match the model input layer
    input_data = np.array([event, time_prev]).reshape((1, max_event_len, 2))
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


def train_tempo_model(epochs=50):
    """Trains an LSTM model to predict the tempo of a piece based on the events and event times."""
    df = pd.read_csv(f"Data\\Tabular\\Soprano.csv", sep=';')
    df = df[['event', 'time', 'tempo']]

    # Normalize the data
    time_scaler = StandardScaler()
    tempo_scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df['tempo'] = df['tempo'].apply(ast.literal_eval).apply(np.array)
    df = df[df['time'].apply(len) > 0]
    df = df[df['event'].apply(len) > 0]
    df = df[df['tempo'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: time_scaler.fit_transform(x.reshape(-1, 1)).flatten())
    df['tempo'] = df['tempo'].apply(lambda x: tempo_scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df[['event', 'time']])
    outputs = np.array(df['tempo'])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x[0]) for x in inputs])
    max_time_len = max([len(x[1]) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int)
                         for x in np.array(df['event'])])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float)
                         for x in np.array(df['time'])])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0.)]).astype(float) for x in outputs])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # LSTM
    model = Sequential()
    model.add(layers.LSTM(64, activation='tanh', input_shape=(max_event_len, 2), return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(64, activation='tanh')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images\\tempo_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Tempo Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Tempo Model MAE", 'MAE')

    # Save the model, scalers, and max length
    model.save(f"Weights\\Tempo\\model.h5")
    pkl.dump(time_scaler, open(f"Weights\\Tempo\\time_scaler.pkl", 'wb'))
    pkl.dump(tempo_scaler, open(f"Weights\\Tempo\\tempo_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights\\Tempo\\seq_len.pkl", 'wb'))

    # Test the model with the first event array; pad the arrays to match the max lengths and then trim after prediction
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time = np.concatenate([df['time'][0], np.full(max_time_len - len(df['time'][0]), 0.)]).astype(float)
    tempo = df['tempo'][0]
    # Reshape the input to match the model input layer
    input_data = np.array([event, time]).reshape((1, max_event_len, 2))
    tempo_pred = model.predict(input_data)
    tempo = tempo_scaler.inverse_transform(tempo.reshape(-1, 1)).flatten()
    tempo_pred = tempo_pred.squeeze()
    tempo_pred = tempo_pred[:len(tempo)]
    tempo_pred = tempo_scaler.inverse_transform(tempo_pred.reshape(-1, 1)).flatten()
    # Convert the tempo to integer arrays
    tempo = np.array([int(x) for x in tempo])
    tempo_pred = np.array([round(x) for x in tempo_pred])
    print("Actual tempo: ", tempo)
    print("Predicted tempo: ", tempo_pred)
    print("Difference: ", tempo - tempo_pred)

    return model, time_scaler, tempo_scaler, max_event_len


def train_time_signature_model(epochs=50):
    """Trains an LSTM model to predict the time signature(s) of a piece based on the events and event times."""
    df = pd.read_csv(f"Data\\Tabular\\Soprano.csv", sep=';')
    df = df[['event', 'time', 'time_signature_count', 'time_signature_beat']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_count'] = df['time_signature_beat'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_beat'] = df['time_signature_beat'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_count'] = df['time_signature_count'].apply(lambda x: np.array([int(y) for y in x]))
    df['time_signature_beat'] = df['time_signature_beat'].apply(lambda x: np.array([int(y) for y in x]))
    df = df[df['time'].apply(len) > 0]
    df = df[df['event'].apply(len) > 0]
    df = df[df['time_signature_count'].apply(len) > 0]
    df = df[df['time_signature_beat'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df[['event', 'time']])
    outputs = np.array(df[['time_signature_count', 'time_signature_beat']])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x) for x in inputs[:, 0]])
    max_time_len = max([len(x) for x in inputs[:, 1]])
    max_output_n_len = max([len(x) for x in outputs[:, 0]])
    max_output_d_len = max([len(x) for x in outputs[:, 1]])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int) for x in inputs[:, 0]])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float) for x in inputs[:, 1]])
    outputs_n = np.array([np.concatenate([x, np.full(max_output_n_len-len(x), 0)]).astype(int) for x in outputs[:, 0]])
    outputs_d = np.array([np.concatenate([x, np.full(max_output_d_len-len(x), 0)]).astype(int) for x in outputs[:, 1]])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.stack((outputs_n, outputs_d), axis=-1)

    # Reshape input and output to time signature per GROUP_SIZE notes; must be a factor of max_event_len (25 works well)
    group_size = 25
    inputs = inputs.reshape((-1, group_size, inputs.shape[-1]))
    outputs = outputs.reshape((-1, group_size, outputs.shape[-1]))

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # LSTM
    model = Sequential()
    model.add(layers.LSTM(50, return_sequences=True, input_shape=[None, 2]))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(2)))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images\\time_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Time Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Time Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights\\TimeSignature\\model.h5")
    pkl.dump(scaler, open(f"Weights\\TimeSignature\\time_sig_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights\\TimeSignature\\seq_len.pkl", 'wb'))

    # Test the model
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time = np.concatenate([df['time'][0], np.full(max_time_len - len(df['time'][0]), 0.)]).astype(float)
    time_sig_counts = df['time_signature_count'][0]
    time_sig_beats = df['time_signature_beat'][0]
    input_data = np.array([event, time]).reshape((1, max_event_len, 2))
    output_data = model.predict(input_data)
    output_data = output_data.squeeze()
    predicted_counts = output_data[:len(time_sig_counts), 0]
    predicted_beats = output_data[:len(time_sig_beats), 1]
    predicted_counts = np.array([round(x) for x in predicted_counts])
    predicted_beats = np.array([round(x) for x in predicted_beats])
    print("Actual counts:", time_sig_counts)
    print("Predicted counts:", predicted_counts)
    print("Difference:", time_sig_counts - predicted_counts)
    print("\nActual beats:", time_sig_beats)
    print("Predicted beats:", predicted_beats)
    print("Difference:", time_sig_beats - predicted_beats)

    return model, scaler, max_event_len


def train_key_model(epochs=10):
    """Trains a Bidirectional LSTM model to predict the key signature(s) of a piece based on the events."""
    df = pd.read_csv(f"Data\\Tabular\\Soprano.csv", sep=';')
    df = df[['event', 'key_signature']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['key_signature'] = df['key_signature'].apply(ast.literal_eval).apply(np.array)
    df['key_signature'] = df['key_signature'].apply(lambda x: np.array([int(y) for y in x]))
    df = df[df['event'].apply(len) > 0]
    df = df[df['key_signature'].apply(len) > 0]
    df['key_signature'] = df['key_signature'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df['event'])
    outputs = np.array(df['key_signature'])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int) for x in inputs])
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0)]).astype(int) for x in outputs])

    # Reshape input and output to key signature per GROUP_SIZE notes; must be a factor of max_event_len (25 works well)
    group_size = 25
    inputs = inputs.reshape((-1, group_size, 1))
    outputs = outputs.reshape((-1, group_size, 1))

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Bi-LSTM
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=[None, 1]))
    model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='tanh')))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images\\key_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Key Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Key Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights\\KeySignature\\model.h5")
    pkl.dump(scaler, open(f"Weights\\KeySignature\\key_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights\\KeySignature\\seq_len.pkl", 'wb'))

    # Test the model
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    key_sigs = df['key_signature'][0]
    input_data = np.array([event]).reshape((1, max_event_len, 1))
    output_data = model.predict(input_data)
    output_data = output_data.squeeze()
    predicted_key_sigs = output_data[:len(key_sigs)]
    key_sigs = scaler.inverse_transform(key_sigs.reshape(-1, 1)).flatten()
    key_sigs = np.array([int(x) for x in key_sigs])
    predicted_key_sigs = scaler.inverse_transform(predicted_key_sigs.reshape(-1, 1)).flatten()
    predicted_key_sigs = np.array([round(x) for x in predicted_key_sigs])
    print("Actual key sigs:", key_sigs)
    print("Predicted key sigs:", predicted_key_sigs)
    print("Difference:", key_sigs - predicted_key_sigs)
    print("\nActual keys:", np.array([key_signature_to_number(x) for x in key_sigs]))
    print("Predicted keys:", np.array([key_signature_to_number(x) for x in predicted_key_sigs]))

    return model, scaler, max_event_len


if __name__ == '__main__':
    print("Hello world!")
    # train_tempo_model(epochs=10)
    # train_time_signature_model(epochs=10)
    # train_key_model(epochs=10)
    train_composition_model("Soprano", epochs=10)
    voices_datasets = ["Soprano", "Alto", "Tenor", "Bass"]
    for voice_dataset in voices_datasets:
        # train_duration_model(voice_dataset, epochs=100)
        pass
    for voice_dataset in voices_datasets:
        # train_composition_model(voice_dataset, epochs=100)
        pass
