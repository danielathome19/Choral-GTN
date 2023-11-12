import music21.clef
from abc import ABC
from tqdm import tqdm
from data_utils import *
from keras.regularizers import l2
from keras import layers, callbacks, models
from keras.optimizers.schedules import LearningRateSchedule


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
    def __init__(self, vocab_size, embed_dim, l2_reg=1e-4):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.l2_reg = l2_reg
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                          embeddings_initializer="he_uniform",
                                          embeddings_regularizer=l2(self.l2_reg))
        self.pos_emb = SinePositionEncoding()

    def call(self, x):
        embedding = self.token_emb(x)
        positions = self.pos_emb(embedding)
        return embedding + positions

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "l2_reg": self.l2_reg})
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, name, num_heads=5, key_dim=256, embed_dim=256, ff_dim=256, dropout_rate=0.3, l2_reg=None):
        super(TransformerBlock, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim, output_shape=self.embed_dim)
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu", kernel_regularizer=l2(l2_reg))
        self.ffn_2 = layers.Dense(self.embed_dim, kernel_regularizer=l2(l2_reg))
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        causal_mask = tf.expand_dims(causal_mask, 1)
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
            "l2_reg": self.l2_reg,
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

    def sample_from(self, probs, temperature):
        if self.top_k == 0:
            probs = probs ** (1 / temperature)
            probs = probs / np.sum(probs)
            return np.random.choice(len(probs), p=probs), probs
        sorted_indices = np.argsort(probs)[::-1]
        top_indices = sorted_indices[:self.top_k]
        top_probs = probs[top_indices]
        top_probs = top_probs ** (1 / temperature)
        top_probs = top_probs / np.sum(top_probs)
        sampled_index = np.random.choice(top_indices, p=top_probs)
        return sampled_index, probs

    def get_note(self, notes, durations, temperature, instrument=None):
        sample_note_idx = 1
        # while sample_note_idx == 1:
        while sample_note_idx == 1 or self.index_to_note[sample_note_idx] == "START":
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

    @staticmethod
    def get_last_attention_layer(model):
        for layer in reversed(model.layers):
            if layer.name.startswith("attention"):
                return layer
        raise ValueError("No attention layer found in the model.")

    def generate(self, start_notes, start_durations, max_tokens, temperature,
                 clef="choral", model=None, intro=False, instrument=None):
        if model is not None:
            self.model = model
        last_attention_layer = self.get_last_attention_layer(self.model)
        attention_model = models.Model(inputs=self.model.input, outputs=last_attention_layer.output)
        # attention_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer("attention").output)
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
                'S': music21.stream.Part(),
                'A': music21.stream.Part(),
                'T': music21.stream.Part(),
                'B': music21.stream.Part()
            }

            clefs = {
                'S': music21.clef.TrebleClef(),
                'A': music21.clef.TrebleClef(),
                'T': music21.clef.Treble8vbClef(),
                'B': music21.clef.BassClef()
            }

            for voice, stream in voice_streams.items():
                stream.append(clefs[voice])

            for sample_token, sample_duration in zip(start_notes, start_durations):
                voice_type = sample_token.split(":")[0]
                new_note = get_choral_midi_note(sample_token, sample_duration)
                if new_note is not None:
                    if voice_type not in ["S", "A", "T", "B"]:
                        voice_streams["S"].append(new_note)
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

            # for _ in tqdm(range(max_tokens * 4), desc="Generating tokens") if self.verbose else range(max_tokens * 4):
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
                        if voice_type not in ["S", "A", "T", "B"]:
                            voice_streams["S"].append(new_note)
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
            start_notes = ["S:START", "A:START", "T:START", "B:START"]
            start_durations = ["0.0", "0.0", "0.0", "0.0"]
            info, midi_stream = self.generate(start_notes, start_durations,
                                              max_tokens=self.generate_len*4, temperature=0.5)
        if self.verbose:
            print(info[-1]["prompt"])
        midi_stream.write("midi", fp=os.path.join(self.output_path, "output-" + str(epoch+1).zfill(4) + ".mid"))


class NoamSchedule(LearningRateSchedule, ABC):
    def __init__(self, d_model, warmup_steps=4000):
        super(NoamSchedule, self).__init__()
        self.d_model = tf.constant(d_model, dtype=tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
