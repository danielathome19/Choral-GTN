import gc
import sys
import ast
import time
import random
import logging
import warnings
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from Transformer import *
from keras import layers
from keras import losses
from functools import partial
from keras import backend as k
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import AdamW
from keras.models import Sequential
from keras_tuner import RandomSearch
from keras.src.utils import plot_model
from keras.callbacks import EarlyStopping
from data_utils import key_signature_to_number
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras_tuner import HyperParameters, Objective, tuners


tf.get_logger().setLevel(logging.ERROR)
k.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def generate_composition(dataset="Combined_choral", generate_len=50, num_to_generate=3, seed_notes=[], seed_durs=[],
                         choral=False, suffix="", temperature=0.5, verify_voices=False):
    DATAPATH = f"Weights/Composition/{dataset}" if not choral else f"Weights/Composition_Choral{suffix}"
    with open(f"{DATAPATH}/{dataset}_notes_vocab.pkl", "rb") as f:
        notes_vocab = pkl.load(f)
    with open(f"{DATAPATH}/{dataset}_durations_vocab.pkl", "rb") as f:
        durations_vocab = pkl.load(f)

    if suffix == "_Transposed2":
        model = build_model(len(notes_vocab), len(durations_vocab), embedding_dim=512, feed_forward_dim=1024,
                            key_dim=64, dropout_rate=0.3, l2_reg=1e-4, num_transformer_blocks=3, num_heads=8)
    else:
        model = build_model(len(notes_vocab), len(durations_vocab), embedding_dim=512, feed_forward_dim=512, key_dim=64,
                            num_heads=8, dropout_rate=0.5, l2_reg=0.0005, num_transformer_blocks=3, gradient_clip=1.5)
    model.load_weights(f"Weights/Composition_Choral{suffix}/checkpoint.ckpt")
    music_gen = MusicGenerator(notes_vocab, durations_vocab, generate_len=generate_len,
                               choral=choral, verbose=True, top_k=30)

    def fail(filename=None):
        os.remove(filename)
        print("Failed to generate piece; retrying...")

    output_filenames = []
    for i in range(num_to_generate):
        while True:
            if not choral:
                info = music_gen.generate(["START"], ["0.0"], max_tokens=generate_len,
                                          temperature=temperature, model=model)
                midi_stream = info[-1]["midi"]  # .chordify()
            else:
                start_notes = ["S:START", "A:START", "T:START", "B:START"]
                start_durations = ["0.0", "0.0", "0.0", "0.0"]
                if len(seed_notes) > 0 and len(seed_durs) > 0:
                    start_notes += seed_notes
                    start_durations += seed_durs
                info, midi_stream = music_gen.generate(start_notes, start_durations, max_tokens=generate_len,
                                                       temperature=temperature, model=model, intro=True)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if not os.path.exists(f"Data/Generated/{dataset}{suffix}"):
                os.makedirs(f"Data/Generated/{dataset}{suffix}")
            filename = os.path.join(f"Data/Generated/{dataset}{suffix}", "output-" + timestr + ".mid")
            midi_stream.write("midi", fp=filename)
            gc.collect()
            # Check the output MIDI file -- if it's less than 0.25 kB, it's probably empty; retry
            if os.path.getsize(filename) < 250:
                fail(filename)
            else:
                if verify_voices:
                    # Load the generated MIDI file and check if it's valid (i.e., has more than 3 tracks)
                    try:
                        mini_gen = music21.converter.parse(filename)
                        if len(mini_gen.parts) < 4:
                            fail(filename)
                            continue
                    except Exception as _:
                        fail(filename)
                        continue
                output_filenames.append(filename)
                break
        gc.collect()
        print(f"Generated piece {i+1}/{num_to_generate}")

    return output_filenames


def build_model(notes_vocab_size, durations_vocab_size, gradient_clip=None,
                embedding_dim=256, feed_forward_dim=256, num_heads=5, key_dim=256, dropout_rate=0.3, l2_reg=1e-4,
                num_transformer_blocks=2, verbose=True):
    note_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    duration_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    note_embeddings = TokenAndPositionEmbedding(notes_vocab_size, embedding_dim // 2, l2_reg=l2_reg)(note_inputs)
    duration_embeddings = TokenAndPositionEmbedding(durations_vocab_size, embedding_dim // 2,
                                                    l2_reg=l2_reg)(duration_inputs)
    embeddings = layers.Concatenate()([note_embeddings, duration_embeddings])
    x = embeddings
    for i in range(num_transformer_blocks):
        x, _ = TransformerBlock(name=f"attention_{i+1}", embed_dim=embedding_dim, ff_dim=feed_forward_dim,
                                num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate, l2_reg=l2_reg)(x)
    note_outputs = layers.Dense(notes_vocab_size, activation="softmax", name="note_outputs",
                                kernel_regularizer=l2(l2_reg))(x)
    duration_outputs = layers.Dense(durations_vocab_size, activation="softmax", name="duration_outputs",
                                    kernel_regularizer=l2(l2_reg))(x)
    model = models.Model(inputs=[note_inputs, duration_inputs], outputs=[note_outputs, duration_outputs])
    lr_schedule = NoamSchedule(embedding_dim)
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=gradient_clip)
    model.compile(optimizer, loss=[losses.SparseCategoricalCrossentropy(), losses.SparseCategoricalCrossentropy()])
    if verbose:
        model.summary()
    return model


def build_model_tuner(hp, notes_vocab_size, durations_vocab_size):
    embedding_dim = hp.Choice('embedding_dim', values=[256, 512, 1024])
    feed_forward_dim = hp.Choice('feed_forward_dim', values=[256, 512, 1024])
    key_dim = hp.Choice('key_dim', values=[64, 128])
    num_heads = hp.Choice('num_heads', values=[4, 8, 12])
    gradient_clip = hp.Choice('gradient_clip', values=[0.5, 1.0, 1.5])
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-3, sampling='LOG')
    num_transformer_blocks = hp.Choice('num_transformer_blocks', values=[1, 2, 3])
    model = build_model(notes_vocab_size, durations_vocab_size, gradient_clip=gradient_clip,
                        embedding_dim=embedding_dim, feed_forward_dim=feed_forward_dim, num_heads=num_heads,
                        key_dim=key_dim, dropout_rate=dropout_rate, l2_reg=l2_reg,
                        num_transformer_blocks=num_transformer_blocks, verbose=False)
    return model


def train_choral_composition_model(epochs=100, suffix="", transposed=False):
    """Trains a choral Transformer model to generate notes and durations."""
    BATCH_SIZE = 128
    GENERATE_LEN = 25
    INCLUDE_AUGMENTED = False
    DATAPATH = "Data/Glob/Combined_choral" if not transposed else "Data/Glob/Combined_transposed"
    VALIDATION_SPLIT = 0.1

    def merge_voice_parts(voice_parts_notes, voice_parts_durations, seq_len=50, max_rest_len=4):
        merged_notes = []
        merged_durations = []
        # Old design: truncate all voice parts to the length of the shortest one (working)
        # min_length = min([len(voice_parts_notes[voice]) for voice in voice_parts_notes])
        # for voice in voice_parts_notes:
        #     voice_parts_notes[voice] = voice_parts_notes[voice][:min_length]
        #     voice_parts_durations[voice] = voice_parts_durations[voice][:min_length]
        # New design attempt 1 -- put notes in cross-voice order ([S, A, T, B, S, A, T, B], ...)
        notes_sequences = {"S": [], "A": [], "T": [], "B": []}
        durations_sequences = {"S": [], "A": [], "T": [], "B": []}
        # for i in range(min_length):
        #     for voice in voice_parts_notes:
        for voice in voice_parts_notes:
            for i in range(len(voice_parts_notes[voice])):
                if max_rest_len is None:
                    notes_sequences[voice] += voice_parts_notes[voice][i].split(" ")
                    durations_sequences[voice] += voice_parts_durations[voice][i].split(" ")
                else:  # Attempt 1.5 -- limit the number of sequential rests
                    split_notes = voice_parts_notes[voice][i].split(" ")
                    split_durations = voice_parts_durations[voice][i].split(" ")
                    rest_cnt = 0
                    for j in range(len(split_notes)):
                        if "rest" in split_notes[j]:
                            rest_cnt += 1
                        else:
                            rest_cnt = 0
                        if rest_cnt <= max_rest_len:
                            notes_sequences[voice].append(split_notes[j])
                            durations_sequences[voice].append(split_durations[j])
                pass
        # # Attempt 1.75 -- truncate to the minimum length after removing rests (not working as intended yet)
        min_length = min([len(notes_sequences[voice]) for voice in notes_sequences])
        for voice in notes_sequences:
            notes_sequences[voice] = notes_sequences[voice][:min_length]
            durations_sequences[voice] = durations_sequences[voice][:min_length]
        note_parts_combined = []
        duration_parts_combined = []
        for i in range(0, min_length * 4, 4):  # each iteration processes one SATB set
            if i + 4 > min_length * 4:  # if we're at the end and there's no full SATB set
                break
            for part in ['S', 'A', 'T', 'B']:
                note_parts_combined.extend(notes_sequences[part][i // 4:i // 4 + 1])
                duration_parts_combined.extend(durations_sequences[part][i // 4:i // 4 + 1])
        # Split the combined sequences into chunks of seq_len
        for i in range(0, len(note_parts_combined), seq_len):
            merged_notes.append(' '.join(note_parts_combined[i:i + seq_len]))
            merged_durations.append(' '.join(duration_parts_combined[i:i + seq_len]))
        return merged_notes, merged_durations

    voices = ["S", "A", "T", "B"]
    voice_parts_notes = {}
    voice_parts_durations = {}
    for voice in voices:
        print(f"Loading {voice} voice parts from {DATAPATH}...")
        voice_parts_notes[voice] = load_pickle_from_slices(f"{DATAPATH}/Combined_{voice}_choral_notes", False)
        voice_parts_durations[voice] = load_pickle_from_slices(f"{DATAPATH}/Combined_{voice}_choral_durations", False)
        if INCLUDE_AUGMENTED:
            for i in range(1, 5):
                aug_notes = load_pickle_from_slices(f"{DATAPATH}/Combined_aug{i}_{voice}_choral_notes", False)
                aug_dur = load_pickle_from_slices(f"{DATAPATH}/Combined_aug{i}_{voice}_choral_durations", False)
                voice_parts_notes[voice] += aug_notes
                voice_parts_durations[voice] += aug_dur

    notes, durations = merge_voice_parts(voice_parts_notes, voice_parts_durations, seq_len=52)  # seq_len=32, 52, 100
    DATARANGE = .25  # May be better to shrink the dataset here rather than after tokenizing
    notes = notes[:int(DATARANGE * len(notes))]
    durations = durations[:int(DATARANGE * len(durations))]
    notes_seq_ds, notes_vectorize_layer, notes_vocab = create_transformer_dataset(notes, BATCH_SIZE)
    durations_seq_ds, durations_vectorize_layer, durations_vocab = create_transformer_dataset(durations, BATCH_SIZE)
    seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

    notes_vocab_size = len(notes_vocab)
    durations_vocab_size = len(durations_vocab)

    # Save vocabularies if they don't exist
    if not os.path.exists(f"Weights/Composition_Choral{suffix}"):
        os.makedirs(f"Weights/Composition_Choral{suffix}")
    if not os.path.exists(f"Weights/Composition_Choral{suffix}/Combined_choral_notes_vocab.pkl"):
        with open(f"Weights/Composition_Choral{suffix}/Combined_choral_notes_vocab.pkl", "wb") as f:
            pkl.dump(notes_vocab, f)
    if not os.path.exists(f"Weights/Composition_Choral{suffix}/Combined_choral_durations_vocab.pkl"):
        with open(f"Weights/Composition_Choral{suffix}/Combined_choral_durations_vocab.pkl", "wb") as f:
            pkl.dump(durations_vocab, f)

    # Create the training set of sequences and the same sequences shifted by one note
    def prepare_inputs(notes, durations):
        notes = tf.expand_dims(notes, -1)
        durations = tf.expand_dims(durations, -1)
        tokenized_notes = notes_vectorize_layer(notes)
        tokenized_durations = durations_vectorize_layer(durations)
        x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
        y = (tokenized_notes[:, 1:], tokenized_durations[:, 1:])
        return x, y

    ds = seq_ds.map(prepare_inputs)  # .shuffle(1024, seed=0) shuffle may be a hindrance # .batch(BATCH_SIZE)

    # Splitting dataset into training and validation
    ds_size = ds.cardinality().numpy()
    train_size = int((1 - VALIDATION_SPLIT) * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    def hyperparameter_search(tuner_trials=15, t_epochs=10, dataset_size=1.0,
                              plot=True, grid_search=False, resume=False, t_suffix=""):
        ptune = partial(build_model_tuner, notes_vocab_size=notes_vocab_size, durations_vocab_size=durations_vocab_size)
        tuner_dir = 'Weights/Hyperparameter_search'
        project_name = 'choral_composition'
        should_overwrite = not resume or not os.path.exists(os.path.join(tuner_dir, project_name))

        if not grid_search:
            tuner = RandomSearch(
                ptune,
                objective=Objective("val_loss", direction="min"),
                max_trials=tuner_trials,
                executions_per_trial=1,
                directory=tuner_dir,
                project_name=project_name,
                overwrite=should_overwrite)
        else:
            tuner = tuners.GridSearch(
                ptune,
                objective=Objective("val_loss", direction="min"),
                directory=tuner_dir,
                project_name=project_name,
                overwrite=should_overwrite)
        train_ds_sm = train_ds.take(int(dataset_size * train_size))
        val_ds_sm = val_ds.take(int(dataset_size * (ds_size - train_size)))
        tuner.search(train_ds_sm, validation_data=val_ds_sm, epochs=t_epochs)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best Hyperparameters: ", best_hp.get_config())
        t_model = tuner.get_best_models(num_models=1)[0]
        t_model.summary()

        if plot:
            trials = tuner.oracle.get_best_trials(num_trials=tuner_trials)
            results = [trial.hyperparameters.values for trial in trials]
            results_df = pd.DataFrame(results)
            results_df['score'] = [trial.score for trial in trials]
            results_df['trial_id'] = [trial.trial_id for trial in trials]
            if not os.path.exists("Logs/HyperparameterSearches"):
                os.makedirs("Logs/HyperparameterSearches")
            results_df.to_csv(f'Logs/HyperparameterSearches/hyperparameter_results{t_suffix}.csv', index=False)

            fig = px.parallel_coordinates(
                results_df,
                color="score",
                labels={col: col for col in results_df.columns},
                color_continuous_scale=px.colors.diverging.Tealrose,
                color_continuous_midpoint=results_df['score'].mean()
            )
            fig.show()
            fig.write_image(f"Images/Hyperparameter_results{t_suffix}.svg", width=1200, height=600)
        return t_model

    # model = hyperparameter_search(grid_search=False, tuner_trials=15, t_epochs=50,
    #                               resume=False, t_suffix="_9", dataset_size=1.0)
    gc.collect()
    # Best Transposed model (.125 [models 5, 10] and .25 [9] datasets);  original key_dim=128
    model = build_model(notes_vocab_size, durations_vocab_size, embedding_dim=512, feed_forward_dim=512, num_heads=8,
                        key_dim=64, dropout_rate=0.000001, l2_reg=1e-6, num_transformer_blocks=3, gradient_clip=1.5)
    # Transposed model (original has 2 transformer blocks, #2 has 3)
    # model = build_model(notes_vocab_size, durations_vocab_size, embedding_dim=512, feed_forward_dim=1024, num_heads=8,
    #                    key_dim=64, dropout_rate=0.3, l2_reg=WEIGHT_DECAY, num_transformer_blocks=3, gradient_clip=1.5)
    plot_model(model, to_file=f'Images/Combined_choral_composition{suffix.lower()}_model.png',
               show_shapes=True, show_layer_names=True, expand_nested=True)

    LOAD_MODEL = False
    if LOAD_MODEL and os.path.exists(f"Weights/Composition_Choral{suffix}"):
        model.load_weights(f"Weights/Composition_Choral{suffix}/checkpoint.ckpt")
        print("Loaded model weights")

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"Weights/Composition_Choral{suffix}/checkpoint.ckpt",
                                                    save_weights_only=True, save_freq="epoch", verbose=0)
    tensorboard_callback = callbacks.TensorBoard(log_dir=f"Logs/Combined_Choral")
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)  # patience=5

    # Tokenize starting prompt
    music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=GENERATE_LEN, choral=True)

    # Train the model
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1,
              callbacks=[checkpoint_callback, early_stopping, tensorboard_callback])  # , music_generator

    model.save(f"Weights/Composition_Choral{suffix}/Combined_choral.keras")

    # Test the model
    TEST_MODEL = False
    if TEST_MODEL:
        start_notes = ["S:START", "A:START", "T:START", "B:START"]
        start_durations = ["0.0", "0.0", "0.0", "0.0"]
        info, midi_stream = music_generator.generate(start_notes, start_durations, max_tokens=50, temperature=0.5)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        midi_stream.write("midi", fp=os.path.join(f"Data/Generated/Combined_choral", "output-" + timestr + ".mid"))

    pass


# Deprecated (for now)
def train_composition_model(dataset="Soprano", epochs=100, load_augmented_dataset=False):
    """Trains a Transformer model to generate notes and durations."""
    PARSE_MIDI_FILES = not os.path.exists(f"Data/Glob/{dataset}_notes.pkl")
    PARSED_DATA_PATH = f"Data/Glob/{dataset}_"
    POLYPHONIC = True
    PLOT_TEST = False
    INCLUDE_AUGMENTED = load_augmented_dataset
    SEQ_LEN = 50
    BATCH_SIZE = 256
    GENERATE_LEN = 50
    WEIGHT_DECAY = 1e-4

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
    with open(f"Weights/Composition/{dataset}/{dataset}_notes_vocab.pkl", "wb") as f:
        pkl.dump(notes_vocab, f)
    with open(f"Weights/Composition/{dataset}/{dataset}_durations_vocab.pkl", "wb") as f:
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

    ds = seq_ds.map(prepare_inputs)  # .repeat(DATASET_REPETITIONS)

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

    # model = build_model(notes_vocab_size, durations_vocab_size, feed_forward_dim=512, num_heads=8)
    model = build_model(notes_vocab_size, durations_vocab_size, embedding_dim=512, feed_forward_dim=1024, num_heads=8,
                        key_dim=64, dropout_rate=0.3, l2_reg=WEIGHT_DECAY, num_transformer_blocks=2, gradient_clip=1.0)
    plot_model(model, to_file=f'Images/{dataset}_composition_model.png',
               show_shapes=True, show_layer_names=True, expand_nested=True)

    LOAD_MODEL = True
    if LOAD_MODEL:
        model.load_weights(f"Weights/Composition/{dataset}/checkpoint.ckpt")
        print("Loaded model weights")

    train_size = int(0.8 * len(notes))
    val_size = len(notes) - train_size
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"Weights/Composition/{dataset}/checkpoint.ckpt",
                                                    save_weights_only=True, save_freq="epoch", verbose=0)
    tensorboard_callback = callbacks.TensorBoard(log_dir=f"Logs/{dataset}")

    # Tokenize starting prompt
    music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=GENERATE_LEN)
    # model.fit(ds, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback, music_generator])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1,
              callbacks=[early_stopping, checkpoint_callback, tensorboard_callback, music_generator])
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


def generate_main():
    DATASET = "Combined_choral"
    num_to_gen = int(input("Enter the number of pieces to generate: ") or 5)
    generate_len = int(input("Enter the length of each piece [around 100-200 works best; 200 by default]: ") or 200)
    temperature = float(input("Enter the temperature to use [0.5-1.0; 0.65 by default]: ") or 0.65)
    suffix = input("Enter the model suffix [_Transposed2, _Transposed13; _Transposed3 by default]: ") or "_Transposed3"
    do_seed = input("Do you want to seed the generation with a specific sequence? [y/n; n by default]: ") or "n"
    if do_seed == "y":
        seed_notes = input("\tEnter notes, alternating SATB (e.g., \"S:C5 A:B-3 T:E4 B:rest S:B4 A:G3 T:F#4 B:F3\"): ")
        seed_durs = input("\tEnter durations (as float, where 1.0 = quarter note; e.g., 4.0 2.0 4.0 1.0 1.0 0.5 ...): ")
        seed_notes = seed_notes.split(" ")
        seed_durs = seed_durs.split(" ")
        if len(seed_notes) != len(seed_durs):
            raise ValueError("Seed notes and durations must be the same length!")
    else:
        seed_notes = []
        seed_durs = []
    output_files = generate_composition(DATASET, generate_len=generate_len, num_to_generate=num_to_gen,
                                        choral=True, suffix=suffix, temperature=temperature,
                                        seed_notes=seed_notes, seed_durs=seed_durs)
    print("Generated the following files:", output_files, "\nPlease post-process the ones you like best.")


if __name__ == '__main__':
    print("Hello, world!")
    generate_main()
    # train_composition_model("Combined", epochs=100, load_augmented_dataset=True)
    # generate_composition("Combined_augmented", num_to_generate=5, generate_len=200, temperature=2.75)
    # train_choral_composition_model(epochs=300, suffix="_Transposed15", transposed=True)
    # generate_composition("Combined_choral", num_to_generate=10, generate_len=200, choral=True,
    #                      temperature=.65, suffix="_Transposed3")
    # for tempr in [0.65, 0.55, 0.45]:  # 0.55, ... , 1.0
    #     generate_composition("Combined_choral", num_to_generate=5, generate_len=200,
    #                          choral=True, temperature=tempr, suffix="_Transposed2")
