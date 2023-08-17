import ast
import sys
import time
import logging
import warnings
import matplotlib.pyplot as plt
from Transformer import *
from keras import layers
from keras import losses
from keras import backend as k
from keras.src.utils import plot_model
from keras.models import Sequential
from data_utils import key_signature_to_number
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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


def generate_composition(dataset="Combined_augmented", generate_len=50, num_to_generate=3):
    with open(f"Weights/Composition/{dataset}_notes_vocab.pkl", "rb") as f:
        notes_vocab = pkl.load(f)
    with open(f"Weights/Composition/{dataset}_durations_vocab.pkl", "rb") as f:
        durations_vocab = pkl.load(f)
    model = build_model(len(notes_vocab), len(durations_vocab), feed_forward_dim=512, num_heads=8)
    model.load_weights(f"Weights/Composition/{dataset}/checkpoint.ckpt")
    music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=generate_len)
    for i in range(num_to_generate):
        while True:
            info = music_generator.generate(["START"], ["0.0"], max_tokens=generate_len,
                                            temperature=0.5, test_model=model)
            midi_stream = info[-1]["midi"].chordify()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(f"Data/Generated/{dataset}", "output-" + timestr + ".mid")
            midi_stream.write("midi", fp=filename)
            # Check the output MIDI file -- if it's less than 1 kB, it's probably empty; retry
            if os.path.getsize(filename) < 1000:
                os.remove(filename)
            else:
                break
        print(f"Generated piece {i+1}/{num_to_generate}")
    pass


def build_model(notes_vocab_size, durations_vocab_size,
                embedding_dim=256, feed_forward_dim=256, num_heads=5, key_dim=256, dropout_rate=0.3):
    note_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    duration_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    note_embeddings = TokenAndPositionEmbedding(notes_vocab_size, embedding_dim // 2)(note_inputs)
    duration_embeddings = TokenAndPositionEmbedding(durations_vocab_size, embedding_dim // 2)(duration_inputs)
    embeddings = layers.Concatenate()([note_embeddings, duration_embeddings])
    x, attention_scores = TransformerBlock(name="attention", embed_dim=embedding_dim, ff_dim=feed_forward_dim,
                                           num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate)(embeddings)
    note_outputs = layers.Dense(notes_vocab_size, activation="softmax", name="note_outputs")(x)  # Attention scores
    duration_outputs = layers.Dense(durations_vocab_size, activation="softmax", name="duration_outputs")(x)
    model = models.Model(inputs=[note_inputs, duration_inputs], outputs=[note_outputs, duration_outputs])
    model.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), losses.SparseCategoricalCrossentropy()])
    model.summary()
    return model


def train_composition_model(dataset="Soprano", epochs=100, load_augmented_dataset=False):
    """Trains a Transformer model to generate notes and times."""
    PARSE_MIDI_FILES = not os.path.exists(f"Data/Glob/{dataset}_notes.pkl")
    PARSED_DATA_PATH = f"Data/Glob/{dataset}_"
    POLYPHONIC = True
    LOAD_MODEL = True
    PLOT_TEST = False
    INCLUDE_AUGMENTED = load_augmented_dataset
    DATASET_REPETITIONS = 1
    SEQ_LEN = 50
    BATCH_SIZE = 256
    GENERATE_LEN = 50

    if dataset != "Combined":
        file_list = glob.glob(f"Data/MIDI/VoiceParts/{dataset}/Isolated/*.mid")
    else:
        file_list = glob.glob(f"Data/MIDI/VoiceParts/{dataset}/*.mid")
    parser = music21.converter

    if PARSE_MIDI_FILES and dataset != "Combined":
        print(f"Parsing {len(file_list)} {dataset} midi files...")
        notes, durations = parse_midi_files(file_list, parser, SEQ_LEN + 1, PARSED_DATA_PATH,
                                            verbose=True, enable_chords=POLYPHONIC, limit=None)
    else:
        if dataset != "Combined":
            notes, durations = load_parsed_files(PARSED_DATA_PATH)
        else:
            notes = load_pickle_from_slices(f"Data/Glob/Combined/Combined_notes", INCLUDE_AUGMENTED)
            durations = load_pickle_from_slices(f"Data/Glob/Combined/Combined_durations", INCLUDE_AUGMENTED)
            if INCLUDE_AUGMENTED:
                dataset += "_augmented"

    example_notes = notes[658]
    # example_durations = durations[658]
    # print("\nNotes string\n", example_notes, "...")
    # print("\nDuration string\n", example_durations, "...")

    notes_seq_ds, notes_vectorize_layer, notes_vocab = create_transformer_dataset(notes, BATCH_SIZE)
    durations_seq_ds, durations_vectorize_layer, durations_vocab = create_transformer_dataset(durations, BATCH_SIZE)
    seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

    # Display the same example notes and durations converted to ints
    example_tokenised_notes = notes_vectorize_layer(example_notes)
    # example_tokenised_durations = durations_vectorize_layer(example_durations)
    # print("{:10} {:10}".format("note token", "duration token"))
    # for i, (note_int, duration_int) in \
    #         enumerate(zip(example_tokenised_notes.numpy()[:11], example_tokenised_durations.numpy()[:11],)):
    #     print(f"{note_int:10}{duration_int:10}")

    notes_vocab_size = len(notes_vocab)
    durations_vocab_size = len(durations_vocab)

    # Save vocabularies
    with open(f"Weights/Composition/{dataset}_notes_vocab.pkl", "wb") as f:
        pkl.dump(notes_vocab, f)
    with open(f"Weights/Composition/{dataset}_durations_vocab.pkl", "wb") as f:
        pkl.dump(durations_vocab, f)

    # # Display some token:note mappings
    # print(f"\nNOTES_VOCAB: length = {len(notes_vocab)}")
    # for i, note in enumerate(notes_vocab[:10]):
    #     print(f"{i}: {note}")
    #
    # print(f"\nDURATIONS_VOCAB: length = {len(durations_vocab)}")
    # # Display some token:duration mappings
    # for i, note in enumerate(durations_vocab[:10]):
    #     print(f"{i}: {note}")

    # Create the training set of sequences and the same sequences shifted by one note
    def prepare_inputs(notes, durations):
        notes = tf.expand_dims(notes, -1)
        durations = tf.expand_dims(durations, -1)
        tokenized_notes = notes_vectorize_layer(notes)
        tokenized_durations = durations_vectorize_layer(durations)
        x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
        y = (tokenized_notes[:, 1:], tokenized_durations[:, 1:])
        return x, y

    ds = seq_ds.map(prepare_inputs).repeat(DATASET_REPETITIONS)

    # example_input_output = ds.take(1).get_single_element()
    # print(example_input_output)

    tpe = TokenAndPositionEmbedding(notes_vocab_size, 32)
    token_embedding = tpe.token_emb(example_tokenised_notes)
    position_embedding = tpe.pos_emb(token_embedding)
    embedding = tpe(example_tokenised_notes)

    def plot_embeddings(in_embedding, title):
        plt.imshow(np.transpose(in_embedding), cmap="coolwarm", interpolation="nearest", origin="lower")
        plt.title(title)
        plt.xlabel("Token")
        plt.ylabel("Embedding Dimension")
        plt.show()

    plot_embeddings(token_embedding, "Token Embedding")
    plot_embeddings(position_embedding, "Position Embedding")
    plot_embeddings(embedding, "Token + Position Embedding")

    model = build_model(notes_vocab_size, durations_vocab_size, feed_forward_dim=512, num_heads=8)
    plot_model(model, to_file=f'Images/{dataset}_composition_model.png',
               show_shapes=True, show_layer_names=True, expand_nested=True)

    if LOAD_MODEL:
        model.load_weights(f"Weights/Composition/{dataset}/checkpoint.ckpt")
        print("Loaded model weights")

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"Weights/Composition/{dataset}/checkpoint.ckpt",
                                                    save_weights_only=True, save_freq="epoch", verbose=0)
    tensorboard_callback = callbacks.TensorBoard(log_dir=f"Logs/{dataset}")

    # Tokenize starting prompt
    music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=GENERATE_LEN)
    model.fit(ds, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback, music_generator])
    model.save(f"Weights/Composition/{dataset}.keras")

    # Test the model
    info = music_generator.generate(["START"], ["0.0"], max_tokens=50, temperature=0.5)
    midi_stream = info[-1]["midi"].chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write("midi", fp=os.path.join(f"Data/Generated/{dataset}", "output-" + timestr + ".mid"))

    if PLOT_TEST:
        max_pitch = 127  # 70
        seq_len = len(info)
        grid = np.zeros((max_pitch, seq_len), dtype=np.float32)

        for j in range(seq_len):
            for i, prob in enumerate(info[j]["note_probs"]):
                try:
                    pitch = music21.note.Note(notes_vocab[i]).pitch.midi
                    grid[pitch, j] = prob
                except:
                    pass

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_yticks([int(j) for j in range(35, 70)])
        plt.imshow(grid[35:70, :], origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5, extent=[0, seq_len, 35, 70])
        plt.title("Note Probabilities")
        plt.xlabel("Timestep")
        plt.ylabel("Pitch")
        plt.show()

        plot_size = 20
        att_matrix = np.zeros((plot_size, plot_size))
        prediction_output = []
        last_prompt = []

        for j in range(plot_size):
            atts = info[j]["atts"].max(axis=0)
            att_matrix[: (j + 1), j] = atts
            prediction_output.append(info[j]["chosen_note"][0])
            last_prompt.append(info[j]["prompt"][0][-1])

        fig, ax = plt.subplots(figsize=(8, 8))
        _ = ax.imshow(att_matrix, cmap="Greens", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, plot_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, plot_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.set_xticks(np.arange(plot_size))
        ax.set_yticks(np.arange(plot_size))
        ax.set_xticklabels(prediction_output[:plot_size])
        ax.set_yticklabels(last_prompt[:plot_size])
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")
        plt.title("Attention Matrix")
        plt.xlabel("Predicted Output")
        plt.ylabel("Last Prompt")
        plt.show()

        pass


def train_duration_model(dataset="Soprano", epochs=100):
    """Trains a Bi-LSTM model to predict the duration of the next note given the previous note and duration."""
    df = pd.read_csv(f"Data/Tabular/{dataset}.csv", sep=';')
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
    plot_model(model, to_file=f'Images/{dataset}_duration_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', f"{dataset} Duration Model Loss (MSE)", 'Loss (MSE)')
    plot_histories(model, 'mae', 'val_mae', f"{dataset} Duration Model MAE", 'MAE')

    # Save the model, scaler, and max length (all 3 max lengths should be the same)
    model.save(f"Weights/Duration/{dataset}_model.h5")
    pkl.dump(scaler, open(f"Weights/Duration/{dataset}_time_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/Duration/{dataset}_seq_len.pkl", 'wb'))

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
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
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
    plot_model(model, to_file=f'Images/tempo_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Tempo Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Tempo Model MAE", 'MAE')

    # Save the model, scalers, and max length
    model.save(f"Weights/Tempo/model.h5")
    pkl.dump(time_scaler, open(f"Weights/Tempo/time_scaler.pkl", 'wb'))
    pkl.dump(tempo_scaler, open(f"Weights/Tempo/tempo_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/Tempo/seq_len.pkl", 'wb'))

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
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
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
    plot_model(model, to_file=f'Images/time_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Time Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Time Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights/TimeSignature/model.h5")
    pkl.dump(scaler, open(f"Weights/TimeSignature/time_sig_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/TimeSignature/seq_len.pkl", 'wb'))

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
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
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
    plot_model(model, to_file=f'Images/key_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Key Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Key Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights/KeySignature/model.h5")
    pkl.dump(scaler, open(f"Weights/KeySignature/key_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/KeySignature/seq_len.pkl", 'wb'))

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
    print("Hello, world!")
    # train_tempo_model(epochs=10)
    # train_time_signature_model(epochs=10)
    # train_key_model(epochs=10)
    # train_composition_model("Combined", epochs=50)
    generate_composition(dataset="Combined", num_to_generate=3, generate_len=100)
    # voices_datasets = ["Soprano", "Bass", "Alto", "Tenor"]
    # for voice_dataset in voices_datasets:
    #     # train_duration_model(voice_dataset, epochs=100)
    #     # train_composition_model(voice_dataset, epochs=100)
    #     pass
