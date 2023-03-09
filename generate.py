import tensorflow as tf
import numpy as np
from utils import midi_to_notes, notes_to_midi, load_midi_files, predict_next_note, mse_with_positive_pressure
from config import NOTE_FEATURES, SEQ_LENGTH, VOCAB_SIZE, TEMPERATURE, NUM_PREDICTIONS
import pretty_midi
import pandas as pd
from random import choice

# load pretrained model
model = tf.keras.models.load_model('test-model.h5', custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})
filenames = load_midi_files()

sample_file = choice(filenames)
pm = pretty_midi.PrettyMIDI(sample_file)
instrument = pm.instruments[0]
raw_notes = midi_to_notes(sample_file)
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)


sample_notes = np.stack([raw_notes[key] for key in NOTE_FEATURES], axis=1)

# The initial sequence of notes; pitch is normalized similar to training
# sequences
input_notes = (sample_notes[:SEQ_LENGTH] / np.array([VOCAB_SIZE, 1, 1]))

generated_notes = []
prev_start = 0
for _ in range(NUM_PREDICTIONS):
    pitch, delta, duration = predict_next_note(input_notes, model, TEMPERATURE)
    start = prev_start + delta
    end = start + duration
    input_note = (pitch, delta, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

generated_notes = pd.DataFrame(generated_notes, columns=(*NOTE_FEATURES, 'start', 'end'))

out_file = 'output.mid'
out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)
