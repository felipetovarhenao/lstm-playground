# import tensorflow as tf
import pretty_midi
import numpy as np
from typing import Iterable
from typing_extensions import Self
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf  # nopep8


class MidiLSTM:
    """
    Polyphonic LSTM model
    """

    NUM_NOTES = 88
    NOTE_OFFSET = 21

    def __init__(self, dataset_dir: str | None = None, max_files: int | None = None, seq_length: int = 25, batch_size: int = 64) -> None:
        self.dataset_dir = dataset_dir
        self.max_files = max_files
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.model = None
        self.learning_rate = 5e-3
        self.epochs = 30
        self.patience = 5
        self.min_delta = 1e-4
        self.dataset = None

    def compile(self) -> None:
        dataset, dataset_size = self.__load_dataset()
        self.dataset = self.__create_training_data(dataset, dataset_size)
        self.__create_model()

    def fit(self):
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=0,
            restore_best_weights=True),
        ]

        self.model.fit(x=self.dataset,
                       epochs=self.epochs,
                       callbacks=callbacks)

    def save(self, path: str = 'model.h5'):
        clean_path = f"{os.path.splitext(path)[0]}.h5"
        self.model.save(clean_path, overwrite=True)

    @classmethod
    def load(cls, path: str) -> Self:
        lstm_model = MidiLSTM()
        lstm_model.model = tf.keras.models.load_model(
            path, custom_objects={'mse_with_positive_pressure': cls.mse_with_positive_pressure})
        return lstm_model

    @staticmethod
    def midi_to_notes(file_path):
        midi = pretty_midi.PrettyMIDI(file_path)
        midi_notes = []
        for instrument in midi.instruments[:1]:
            sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
            current_start = sorted_notes[0].start
            chord = [sorted_notes[0].pitch]
            for note in sorted_notes:
                if note.start > current_start:
                    midi_notes.append(MidiLSTM.encode_chord(chord))
                    current_start = note.start
                    chord = [note.pitch]
                else:
                    chord.append(note.pitch)
        return np.array(midi_notes)

    def notes_to_midi(notes: np.ndarray, out_file: str = 'out.mid', tempo: int = 120):
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

        start = 0
        for pitch, delta, duration in notes:
            end = float(start + duration)
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(pitch),
                start=start,
                end=end,
            )
            start += delta
            instrument.notes.append(note)

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm

    @classmethod
    def encode_chord(cls, notes: Iterable) -> np.ndarray:
        ohe = np.zeros(shape=(cls.NUM_NOTES,))
        ohe[np.array(notes).astype('int32') - cls.NOTE_OFFSET] = 50.0
        return ohe

    @classmethod
    def decode_chord(cls, ohe: np.ndarray) -> np.ndarray:
        notes = np.argwhere(ohe > 0.0) + cls.NOTE_OFFSET
        return sorted(notes.reshape((notes.shape[0], )))

    def __load_dataset(self) -> tuple:
        all_notes = []
        filenames = os.listdir(self.dataset_dir)
        if self.max_files:
            filenames = filenames[:self.max_files]
        for file in filenames:
            file_path = os.path.join(self.dataset_dir, file)
            midi_notes = MidiLSTM.midi_to_notes(file_path)
            all_notes.extend(midi_notes)
        return tf.data.Dataset.from_tensor_slices(np.array(all_notes)), len(all_notes)

    def __create_training_data(self,
                               dataset: tf.data.Dataset,
                               n_notes: int) -> tf.data.Dataset:

        # add 1 extra for the final note event, to be used as label
        seq_length = self.seq_length + 1

        # shift == hop size between windows: e.g., [0, 1], [1, 2] ...
        # stride == hop size within windows: e.g., [0, 2], [1, 3], ...
        windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

        def flatten(x):
            return x.batch(seq_length, drop_remainder=True)

        # flat_map flattens the" dataset of datasets" into a dataset of tensors
        sequences = windows.flat_map(flatten)

        # separate last notes and create a label from it
        def split_labels(sequence):
            input_seq = sequence[:-1]
            output_note = sequence[-1]
            return input_seq, {'pitch': output_note}

        sequences = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
        buffer_size = n_notes - seq_length  # the number of items in the dataset

        return (sequences
                .shuffle(buffer_size)
                .batch(self.batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    def __create_model(self) -> None:
        input_shape = (self.seq_length, MidiLSTM.NUM_NOTES)
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(units=128, dropout=0.25)(inputs)

        outputs = {
            'pitch': tf.keras.layers.Dense(units=MidiLSTM.NUM_NOTES, name='pitch')(x)
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            'pitch': MidiLSTM.mse_with_positive_pressure
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def summary(self):
        self.model.summary()

    @classmethod
    def mse_with_positive_pressure(cls, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ Custom loss function to enforce positive values for note delta and duration """
        mse = (y_true - y_pred) ** 2

        # penalize negative values for y_pred, which get multiplied by 10, significantly increasing the error
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)

    def predict_next_notes(self,
                           notes: np.ndarray) -> int:
        """Generates a note IDs using a trained sequence model."""

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)
        predictions = self.model.predict(inputs)
        pitch_multihot = predictions['pitch']
        pitch_multihot = tf.squeeze(pitch_multihot, 0)
        pitch_multihot = np.clip(pitch_multihot / 10.0, 0.0, 1.0)

        return np.round(pitch_multihot)

    def generate_notes(self, initial_seq: np.ndarray, num_predictions: int = 100) -> np.ndarray:
        input_notes = [self.encode_chord(x) if len(x) != MidiLSTM.NUM_NOTES else x for x in initial_seq]
        generated_notes = []
        for _ in range(num_predictions):
            encoded_chord = self.predict_next_notes(input_notes)
            generated_notes.append(encoded_chord)

            # shift input notes array
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(encoded_chord, 0), axis=0)
        return generated_notes

    def generate_midi(self, prompt: np.ndarray | None = None, out_file: str = 'out.mid') -> None:
        prompt = MidiLSTM.midi_to_notes('data/bach_inventions/bach_inventions_772_[free]_(c)simonetto.mid')
        midi_sequence = self.generate_notes(initial_seq=prompt[:self.seq_length])
        MidiLSTM.notes_to_midi(midi_sequence, out_file=out_file)

    @classmethod
    def notes_to_midi(cls, encoded_chords: np.ndarray, out_file: str = 'out.mid', tempo: int = 120):
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
        start = 0
        for encoded_chord in encoded_chords:
            chord = cls.decode_chord(encoded_chord)
            end = float(start + 0.125)
            for pitch in chord:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(pitch),
                    start=start,
                    end=end,
                )
                instrument.notes.append(note)
            start += 0.125

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm
