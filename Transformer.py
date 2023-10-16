import music21.clef
from data_utils import *
from keras import layers, callbacks, models


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)
    return tf.tile(mask, mult)


class SinePositionEncoding(layers.Layer):
    def __init__(self, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        seq_length = input_shape[-2]
        hidden_dim = input_shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(min_freq, tf.cast(2 * (tf.range(hidden_dim) // 2),
                                              self.compute_dtype) / tf.cast(hidden_dim, self.compute_dtype))
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # Even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_dim) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # Embedding shape is [seq_length, hidden_size]
        positional_encodings = (tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask)
        return tf.broadcast_to(positional_encodings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"max_wavelength": self.max_wavelength,})
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                          embeddings_initializer="he_uniform")
        self.pos_emb = SinePositionEncoding()

    def call(self, x):
        embedding = self.token_emb(x)
        positions = self.pos_emb(embedding)
        return embedding + positions

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim,})
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, name, num_heads=5, key_dim=256, embed_dim=256, ff_dim=256, dropout_rate=0.3):
        super(TransformerBlock, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        # self.attn = layers.MultiHeadAttention(num_heads, key_dim, output_shape=embed_dim)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim, output_shape=self.embed_dim)
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        causal_mask = tf.expand_dims(causal_mask, 1)  # Add an additional dimension; TODO: remove
        attention_output, attention_scores = self.attn(inputs, inputs, attention_mask=causal_mask,
                                                       return_attention_scores=True)
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return self.ln_2(out1 + ffn_output), attention_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class MusicGenerator(callbacks.Callback):
    def __init__(self, index_to_note, index_to_duration, top_k=10, generate_len=50,
                 output_path="Data/Generated/Training", choral=False, verbose=False):
        super().__init__()
        self.index_to_note = index_to_note
        self.note_to_index = {note: index for index, note in enumerate(index_to_note)}
        self.index_to_duration = index_to_duration
        self.duration_to_index = {duration: index for index, duration in enumerate(index_to_duration)}
        self.top_k = top_k
        self.generate_len = generate_len
        self.output_path = output_path
        self.verbose = verbose
        self.choral = choral

    @staticmethod
    def sample_from(probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def get_note(self, notes, durations, temperature, instrument=None):
        sample_note_idx = 1
        while sample_note_idx == 1:
            sample_note_idx, note_probs = self.sample_from(notes[0][-1], temperature)
            sample_note = self.index_to_note[sample_note_idx]

        sample_duration_idx = 1
        while sample_duration_idx == 1:
            sample_duration_idx, duration_probs = self.sample_from(durations[0][-1], temperature)
            sample_duration = self.index_to_duration[sample_duration_idx]

        new_note = get_midi_note(sample_note, sample_duration, instrument) if not self.choral else \
                   get_choral_midi_note(sample_note, sample_duration)
        return (
            new_note,
            sample_note_idx,
            sample_note,
            note_probs,
            sample_duration_idx,
            sample_duration,
            duration_probs,
        )

    def generate(self, start_notes, start_durations, max_tokens, temperature,
                 clef="choral", model=None, intro=False, instrument=None):
        if model is not None:
            self.model = model
        attention_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer("attention").output)
        start_note_tokens = [self.note_to_index.get(x, 1) for x in start_notes]
        start_duration_tokens = [self.duration_to_index.get(x, 1) for x in start_durations]
        sample_note = None
        sample_duration = None
        info = []

        if not self.choral:
            midi_stream = music21.stream.Stream()

            if clef == "treble":
                midi_stream.append(music21.clef.TrebleClef())
            elif clef == "bass":
                midi_stream.append(music21.clef.BassClef())
            elif clef == "tenor":
                midi_stream.append(music21.clef.Treble8vbClef())
            elif clef == "choral":
                midi_stream.append(music21.clef.TrebleClef())
                midi_stream.append(music21.clef.BassClef())

            if instrument is not None:
                instruments = {"Soprano": music21.instrument.Soprano(), "Alto": music21.instrument.Alto(),
                               "Tenor": music21.instrument.Tenor(), "Bass": music21.instrument.Bass()}
                midi_stream.append(instruments[instrument])

            for sample_note, sample_duration in zip(start_notes, start_durations):
                new_note = get_midi_note(sample_note, sample_duration, instrument)
                if new_note is not None:
                    midi_stream.append(new_note)

            if intro:
                info.append({
                    "prompt": [start_notes.copy(), start_durations.copy()],
                    "midi": midi_stream,
                    "chosen_note": (sample_note, sample_duration),
                    "note_probs": 1,
                    "duration_probs": 1,
                    "atts": [],
                })

            while len(start_note_tokens) < max_tokens:
                x1 = np.array([start_note_tokens])
                x2 = np.array([start_duration_tokens])
                notes, durations = self.model.predict([x1, x2], verbose=0)

                repeat = True
                while repeat:
                    (
                        new_note,
                        sample_note_idx,
                        sample_note,
                        note_probs,
                        sample_duration_idx,
                        sample_duration,
                        duration_probs,
                    ) = self.get_note(notes, durations, temperature, instrument)

                    if (isinstance(new_note, music21.chord.Chord) or isinstance(new_note, music21.note.Note) or
                        isinstance(new_note, music21.note.Rest)) and sample_duration == "0.0":
                        repeat = True
                        continue
                    elif intro and (isinstance(new_note, music21.tempo.MetronomeMark) or
                                    isinstance(new_note, music21.key.Key) or
                                    isinstance(new_note, music21.meter.TimeSignature)):
                        repeat = True
                        continue
                    else:
                        repeat = False

                if new_note is not None:
                    midi_stream.append(new_note)

                _, att = attention_model.predict([x1, x2], verbose=0)

                info.append({
                    "prompt": [start_notes.copy(), start_durations.copy()],
                    "midi": midi_stream,
                    "chosen_note": (sample_note, sample_duration),
                    "note_probs": note_probs,
                    "duration_probs": duration_probs,
                    "atts": att[0, :, -1, :],
                })
                start_note_tokens.append(sample_note_idx)
                start_duration_tokens.append(sample_duration_idx)
                start_notes.append(sample_note)
                start_durations.append(sample_duration)

                if sample_note == "START":
                    break

            return info
        else:
            voice_streams = {
                'Soprano': music21.stream.Part(),
                'Alto': music21.stream.Part(),
                'Tenor': music21.stream.Part(),
                'Bass': music21.stream.Part()
            }

            clefs = {
                'Soprano': music21.clef.TrebleClef(),
                'Alto': music21.clef.TrebleClef(),
                'Tenor': music21.clef.Treble8vbClef(),
                'Bass': music21.clef.BassClef()
            }

            for voice, stream in voice_streams.items():
                stream.append(clefs[voice])

            for sample_token, sample_duration in zip(start_notes, start_durations):
                voice_type = sample_token.split(":")[0]
                new_note = get_choral_midi_note(sample_token, sample_duration)
                if new_note is not None:
                    if voice_type not in ["Soprano", "Alto", "Tenor", "Bass"]:
                        voice_streams["Soprano"].append(new_note)
                    else:
                        voice_streams[voice_type].append(new_note)

            if intro:
                info.append({
                    "prompt": [start_notes.copy(), start_durations.copy()],
                    "midi": voice_streams,
                    "chosen_note": (sample_note, sample_duration),
                    "note_probs": 1,
                    "duration_probs": 1,
                    "atts": [],
                })

            while len(start_note_tokens) < max_tokens * 4:
                x1 = np.array([start_note_tokens])
                x2 = np.array([start_duration_tokens])
                notes, durations = self.model.predict([x1, x2], verbose=0)

                repeat = True
                while repeat:
                    (
                        new_note,
                        sample_note_idx,
                        sample_note,
                        note_probs,
                        sample_duration_idx,
                        sample_duration,
                        duration_probs,
                    ) = self.get_note(notes, durations, temperature)

                    voice_type = sample_note.split(":")[0]

                    if (isinstance(new_note, music21.chord.Chord) or isinstance(new_note, music21.note.Note) or
                        isinstance(new_note, music21.note.Rest)) and sample_duration == "0.0":
                        repeat = True
                        continue
                    elif intro and (isinstance(new_note, music21.tempo.MetronomeMark) or
                                    isinstance(new_note, music21.key.Key) or
                                    isinstance(new_note, music21.meter.TimeSignature)):
                        repeat = True
                        continue
                    else:
                        repeat = False

                    if new_note is not None:
                        if voice_type not in ["Soprano", "Alto", "Tenor", "Bass"]:
                            voice_streams["Soprano"].append(new_note)
                        else:
                            voice_streams[voice_type].append(new_note)

                _, att = attention_model.predict([x1, x2], verbose=0)

                info.append({
                    "prompt": [start_notes.copy(), start_durations.copy()],
                    "midi": voice_streams,
                    "chosen_note": (sample_note, sample_duration),
                    "note_probs": note_probs,
                    "duration_probs": duration_probs,
                    "atts": att[0, :, -1, :],
                })
                start_note_tokens.append(sample_note_idx)
                start_duration_tokens.append(sample_duration_idx)
                start_notes.append(sample_note)
                start_durations.append(sample_duration)

                if sample_note == "START":
                    break

            midi_stream = music21.stream.Score()
            for voice, stream in voice_streams.items():
                midi_stream.insert(0, stream)

            return info, midi_stream

    def on_epoch_end(self, epoch, logs=None):
        if not self.choral:
            info = self.generate(["START"], ["0.0"], max_tokens=self.generate_len, temperature=0.5)
            midi_stream = info[-1]["midi"].chordify()
        else:
            start_notes = ["Soprano:START", "Alto:START", "Tenor:START", "Bass:START"]
            start_durations = ["0.0", "0.0", "0.0", "0.0"]
            info, midi_stream = self.generate(start_notes, start_durations,
                                              max_tokens=self.generate_len*4, temperature=0.5)
        if self.verbose:
            print(info[-1]["prompt"])
        midi_stream.write("midi", fp=os.path.join(self.output_path, "output-" + str(epoch+1).zfill(4) + ".mid"))


# TODO: fix or remove
class TokenAndPositionEmbedding2(layers.Layer):
    def __init__(self, max_values, embed_dim):
        super(TokenAndPositionEmbedding2, self).__init__()
        self.embed_dim = embed_dim
        self.embeddings = [layers.Embedding(input_dim=v, output_dim=embed_dim) for v in max_values]
        self.pos_emb = SinePositionEncoding()

    def call(self, x):
        # Assuming x is of shape [batch_size, seq_len, num_features]
        embeddings = [embed(x[..., i]) for i, embed in enumerate(self.embeddings)]
        embedding = tf.reduce_sum(embeddings, axis=0)  # Combine embeddings
        positions = self.pos_emb(embedding)
        return embedding + positions

    def get_config(self):
        config = super().get_config()
        config.update({"max_values": [e.input_dim for e in self.embeddings], "embed_dim": self.embed_dim})
        return config


class MusicGenerator2(callbacks.Callback):
    def __init__(self, token_to_index, scaler=None, top_k=10, generate_len=50,
                 output_path="Data/Generated/Training", verbose=False):
        super().__init__()
        self.token_to_index = token_to_index
        self.index_to_token = {index: token for token, index in token_to_index.items()}
        self.top_k = top_k
        self.generate_len = generate_len
        self.output_path = output_path
        self.scaler = scaler
        self.verbose = verbose

    @staticmethod
    def sample_from(probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def get_token(self, tokens, temperature):
        sample_token_idx = 1
        while sample_token_idx == 1:
            sample_token_idx, token_probs = self.sample_from(tokens[0][-1], temperature)
            sample_token = self.index_to_token[sample_token_idx]
        return sample_token, sample_token_idx, token_probs

    def generate(self, start_tokens, max_tokens, temperature, model=None):
        if model is not None:
            self.model = model

        start_token_indices = [self.token_to_index.get(x, 1) for x in start_tokens]
        sample_token = None
        info = []

        midi_stream = self.tokens_to_midi(start_tokens)  # Generate initial midi stream from start tokens

        while len(start_token_indices) < max_tokens:
            x1 = np.array([start_token_indices])
            preds = self.model.predict(x1, verbose=0)[0]

            # Your sampling logic here. For simplicity, let’s use argmax.
            sample_token_idx = np.argmax(preds[-1])
            sample_token = self.index_to_token[sample_token_idx]

            # If you have an end token, you might break the loop here.
            if sample_token == 'END':
                break

            # Add the new token to the sequence and generate its midi.
            start_tokens.append(sample_token)
            start_token_indices.append(sample_token_idx)
            new_midi_part = self.tokens_to_midi([sample_token])
            midi_stream.append(new_midi_part)

        return midi_stream

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [0, -1, None, None, 0]
        midi_stream = self.generate(start_tokens, max_tokens=self.generate_len, temperature=0.5)
        if self.verbose:
            print("Generated tokens:", start_tokens)
        midi_stream.write("midi", fp=os.path.join(self.output_path, "output-" + str(epoch+1).zfill(4) + ".mid"))

    def tokens_to_midi(self, tokens):
        if self.scaler is not None:
            tokens = self.scaler.inverse_transform(tokens)
        midi_stream = music21.stream.Score()
        for token in tokens:
            music21_obj, duration_obj = decode_token_to_music21(token)
            if music21_obj is not None:
                if duration_obj is not None:
                    music21_obj.duration = duration_obj
                midi_stream.append(music21_obj)
        return midi_stream
