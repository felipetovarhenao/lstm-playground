from music_rnn.model import MidiLSTM

model = MidiLSTM.load('new-model.h5')
model.generate_midi(out_file='new_test.mid')
