from bach_rnn.config import (
    DATASET_DIR,
    MIN_DELTA,
    EPOCHS,
    PATIENCE
)
from bach_rnn.utils import (
    load_midi_dataset,
    create_training_data,
    get_lstm_model
)
import tensorflow as tf

midi_dataset, n_notes = load_midi_dataset(DATASET_DIR, 1)
training_data = create_training_data(midi_dataset, n_notes=n_notes)

model = get_lstm_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        verbose=0,
        restore_best_weights=True),
]

history = model.fit(x=training_data,
                    epochs=EPOCHS,
                    callbacks=callbacks)

model.save('bach-model.h5',)
