import numpy as np
import pandas as pd
from config import NOTE_FEATURES, VOCAB_SIZE, BATCH_SIZE, SEQ_LENGTH, EPOCHS, LEARNING_RATE, NUM_FILES, MIN_DELTA
from utils import create_sequences, midi_to_notes, mse_with_positive_pressure, load_midi_files
from random import shuffle
import tensorflow as tf  # nopep8

# get list of all midi files
filenames = load_midi_files()
shuffle(filenames)

# create pandas data frames with note sequences for each MIDI file
all_notes = [midi_to_notes(f) for f in filenames[:NUM_FILES]]

# concat data frames into a single one
all_notes = pd.concat(all_notes)
n_notes = len(all_notes)

# create training dataset as numpy array
train_notes = np.stack([all_notes[key] for key in NOTE_FEATURES], axis=1)

# convert array to tensorflow dataset
notes_dataset = tf.data.Dataset.from_tensor_slices(train_notes)


sequence_dataset = create_sequences(notes_dataset, SEQ_LENGTH, VOCAB_SIZE)

buffer_size = n_notes - SEQ_LENGTH  # the number of items in the dataset

train_data = (sequence_dataset
              .shuffle(buffer_size)
              .batch(BATCH_SIZE, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

# ---------------------------
# CREATE MODEL
# ---------------------------
input_shape = (SEQ_LENGTH, 3)

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(units=128)(inputs)

outputs = {
    'pitch': tf.keras.layers.Dense(units=128, name='pitch')(x),
    'delta': tf.keras.layers.Dense(units=1, name='delta')(x),
    'duration': tf.keras.layers.Dense(units=1, name='duration')(x),
}

model = tf.keras.Model(inputs, outputs)

loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'delta': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss=loss,
              optimizer=optimizer,
              # use smaller weight for pitch, given domain difference
              loss_weights={
                  'pitch': 1.0 / VOCAB_SIZE,
                  'delta': 1.0,
                  'duration': 1.0,
              })

# losses = model.evaluate(train_data, return_dict=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=MIN_DELTA,
        patience=5,
        verbose=1,
        restore_best_weights=True),
]

history = model.fit(x=train_data,
                    epochs=EPOCHS,
                    callbacks=callbacks)
print(history)
# save model
model.save('test-model.h5',)
