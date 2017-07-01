import codecs
import pandas as pd

data_path = "../data/socialmedia-disaster-tweets.csv"
df = pd.read_csv(data_path, encoding='utf8')
df = df[['text', 'choose_one']]
df = df[df.choose_one.isin(['Relevant', 'Not Relevant'])]
df['int_label'] = 0
df.loc[df.choose_one == 'Relevant', 'int_label'] = 1
# Text Processing
X_text = [t.encode('utf8') for t in df.text.tolist()]
print X_text[:10]

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

y = df.int_label.values

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)
X_padded = pad_sequences(sequences)
print "Shape of input sequences:", X_padded.shape
print "First element of data:", X_padded[0]

print tokenizer.word_index.keys()[:10]
print type(tokenizer.word_index.keys()[0])

import json
with codecs.open("word_index.json", encoding="utf8", mode="w") as outfile:
    json.dump(tokenizer.word_index, outfile)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index), output_dim=100))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Different loss

model.fit(X_padded, y, epochs=5, batch_size=32)
model.save("../data/lstm.h5")



