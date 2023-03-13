from music_rnn.model import MidiLSTM

# dataset

model = MidiLSTM(dataset_dir='data/bach_inventions', max_files=5)
model.compile()
model.fit()
model.save('new-model.h5')
