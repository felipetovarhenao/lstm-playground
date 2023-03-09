from config import NOTE_FEATURES

import tensorflow as tf
import pandas as pd
import collections
import pretty_midi
import numpy as np
import pathlib
import glob


def create_sequences(dataset: tf.data.Dataset,
                     seq_length: int = 25,
                     vocab_size: int = 128) -> tf.data.Dataset:
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

    # Normalize pitch for all notes in sequence — X.shape = (seq_length, 3)
    def scale_pitch(X):
        X = X / [vocab_size, 1.0, 1.0]
        return X

    # separate last notes and create a label from it
    def split_labels(sequence):
        inputs = sequence[:-1]
        labels_dense = sequence[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(NOTE_FEATURES)}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """ Creates a pandas data frame from a midi file path """
    midi = pretty_midi.PrettyMIDI(midi_file)

    # extract only first instrument, since dataset is all midi piano
    instrument = midi.instruments[0]
    notes = collections.defaultdict(list)

    # sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['delta'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({col_name: np.array(col_value) for col_name, col_value in notes.items()})


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    """ Custom loss function to enforce positive values for note delta and duration """
    mse = (y_true - y_pred) ** 2
    # punishes negative values for y_pred, which get multiplied by 10, significantly increasing the error
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def load_midi_files():
    data_dir = pathlib.Path('data/maestro-v2.0.0')

    # create directory
    if not data_dir.exists():

        # disable ssl verification
        import ssl
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context

        # download maestro dataset zip file and extract
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip', extract=True,
            cache_dir='.', cache_subdir='data')

    # return list of all midi files
    return glob.glob(str(data_dir/'**/*.mid*'))


def notes_to_midi(notes: pd.DataFrame,
                  out_file: str,
                  instrument_name: str,
                  velocity: int = 100,  # note loudness
                  ) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note['delta'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def predict_next_note(notes: np.ndarray,
                      model: tf.keras.Model,
                      temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    delta = predictions['delta']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    delta = tf.squeeze(delta, axis=-1)

    # `delta` and `duration` values should be non-negative
    delta = tf.maximum(0, delta)
    duration = tf.maximum(0, duration)

    return int(pitch), float(delta), float(duration)
