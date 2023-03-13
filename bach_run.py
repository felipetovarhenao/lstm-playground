from bach_rnn.utils import load_model, midi_to_notes, generate_notes, notes_to_midi
from bach_rnn.config import SEQ_LENGTH


model = load_model('bach-model.h5')

initial_notes = midi_to_notes('./data/bach_inventions/bach_inventions_772_[free]_(c)simonetto.mid')
notes = generate_notes(model, initial_notes)
# notes = initial_notes
midi = notes_to_midi(notes, 'output.mid')
