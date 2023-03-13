from .config import SEQ_LENGTH, LEARNING_RATE, VOCAB_SIZE, BATCH_SIZE
from typing import Iterable
import os
import pretty_midi
import numpy as np
import tensorflow as tf


def get_rhythmic_units(max_den: int = 16):
    units = set()
    for den in range(1, max_den):
        for num in range(den):
            val = quantize(num/den, 1/max_den)
            frac = val.as_integer_ratio()
            if any([x > max_den for x in frac]):
                continue
            units.add(val)
    return np.array(list(units))


def quantize(value: float, quantum: float = 0.1):
    return round(value/quantum) * quantum


RHYTHMIC_UNITS = get_rhythmic_units()


def beat_unit_to_duration(beat_unit: float, tempo: float):
    return (240.0 / tempo) * beat_unit


def duration_to_beat_unit(duration: float, tempo: float):
    return duration / (240.0 / tempo)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def midi_to_notes(file_path):
    midi = pretty_midi.PrettyMIDI(file_path)
    midi_notes = []
    tempi_markers, tempi = midi.get_tempo_changes()
    num_tempi = len(tempi)
    for instrument in midi.instruments[:1]:
        sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
        tempo_index = 0
        current_tempo, next_tempo_marker = tempi[tempo_index], tempi_markers[min(1, num_tempi - 1)]
        for i, note in enumerate(sorted_notes[:-1]):
            start = note.start
            next_start = sorted_notes[i+1].start

            delta = next_start - start

            note_vector = np.array([start,
                                    note.pitch,
                                    delta,
                                    note.duration,
                                    # note.velocity,
                                    # instrument.program
                                    ])
            midi_notes.append(note_vector)
            if next_tempo_marker and start >= next_tempo_marker:
                tempo_index += 1
                current_tempo = tempi_markers[tempo_index]
                next_tempo_marker = tempi_markers[tempo_index + 1] if next_tempo_marker < num_tempi - 1 else None
    midi_notes.sort(key=lambda n: (n[0], n[1]))
    return np.array(midi_notes)[:, 1:]


def load_midi_dataset(path, max_files: int | None = None) -> Iterable:
    """
    Returns concatenated notes from bach dataset, with the following numpy array formatting:
    [midi_pitch, inter_onset_duration, duration, velocity, instrument_program]
    """
    all_notes = []
    filenames = os.listdir(path)
    if max_files:
        filenames = filenames[:max_files]
    for file in filenames:
        file_path = os.path.join(path, file)
        midi_notes = midi_to_notes(file_path)
        all_notes.extend(midi_notes)
    return tf.data.Dataset.from_tensor_slices(np.array(all_notes)), len(all_notes)


def create_training_data(dataset: tf.data.Dataset,
                         n_notes: int,
                         seq_length: int = SEQ_LENGTH,
                         batch_size: int = BATCH_SIZE,
                         vocab_size: int = VOCAB_SIZE) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""

    # add 1 extra for the final note event, to be used as label
    seq_length = seq_length + 1

    # shift == hop size between windows: e.g., [0, 1], [1, 2] ...
    # stride == hop size within windows: e.g., [0, 2], [1, 3], ...
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    def flatten(x):
        return x.batch(seq_length, drop_remainder=True)

    # flat_map flattens the" dataset of datasets" into a dataset of tensors
    sequences = windows.flat_map(flatten)

    # Normalize pitch for all notes in sequence
    def normalize_pitch(X, norm):
        X = X / norm
        return X

    # separate last notes and create a label from it
    def split_labels(sequence):
        input_seq = sequence[:-1]
        output_note = sequence[-1]
        norm = [vocab_size] + ([1.0] * (output_note.shape[-1] - 1))
        labels = {key: output_note[i] for i, key in enumerate(['pitch', 'delta', 'duration'])}
        return normalize_pitch(input_seq, norm), labels

    sequences = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    buffer_size = n_notes - seq_length  # the number of items in the dataset

    return (sequences
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    """ Custom loss function to enforce positive values for note delta and duration """
    mse = (y_true - y_pred) ** 2

    # penalize negative values for y_pred, which get multiplied by 10, significantly increasing the error
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def get_lstm_model() -> tf.keras.Model:
    input_shape = (SEQ_LENGTH, 3)
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(units=VOCAB_SIZE, dropout=0.5, recurrent_dropout=0.1)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(units=VOCAB_SIZE, name='pitch')(x),
        'delta': tf.keras.layers.Dense(units=1, name='delta')(x),
        'duration': tf.keras.layers.Dense(units=1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # 'delta': mse_with_positive_pressure,
        # 'duration': mse_with_positive_pressure,
        'delta': tf.keras.losses.MeanSquaredError(),
        'duration': tf.keras.losses.MeanSquaredError(),
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  loss_weights=[
                      1.0 / VOCAB_SIZE,
                      1.0,
                      1.0
                  ])
    return model


def load_model(path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})


def predict_next_note(notes: np.ndarray,
                      model: tf.keras.Model,
                      temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)

    pitch_logits = predictions['pitch']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)

    duration = predictions['duration']
    duration = tf.squeeze(duration, axis=-1)
    duration = tf.maximum(0, duration)

    delta = predictions['delta']
    delta = tf.squeeze(delta, axis=-1)
    delta = tf.maximum(0, delta)

    return int(pitch), float(delta), float(duration)


def generate_notes(model: tf.keras.models.Model,
                   initial_seq: np.ndarray,
                   num_predictions: int = 100,
                   temperature: float = 1.0) -> np.ndarray:
    input_notes = (initial_seq[:SEQ_LENGTH] / np.array([VOCAB_SIZE, 1, 1]))
    generated_notes = []
    for _ in range(num_predictions):
        pitch, delta, duration = predict_next_note(input_notes, model, temperature)
        input_note = (pitch, delta, duration)
        generated_notes.append(input_note)

        # shift input notes array
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    return generated_notes


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
