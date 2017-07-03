from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet, Label
from bokeh.io import output_notebook

import json
from keras.models import load_model

def get_embedding_weights(model, word_index):
    embedding_weights = model.layers[0].get_weights()[0]
    print embedding_weights.shape
    print embedding_weights.max()
    print embedding_weights.min()

    output_notebook()
        
    random_words = np.random.choice(word_index.keys(), size=600, replace=False)
    random_embeddings = np.array([embedding_weights[word_index[rw]] for rw in random_words])
    print random_embeddings.shape
    random_words[:10]
    return random_words, random_embeddings

def load_word_index():
    print "Loading word index from trained POS tagger"
    with open('../data/pos_word_index.json', 'r') as infile:
        word_index = json.load(infile)
    return word_index

def plot_embeddings(embedding_df):
    words = embedding_df.words.tolist()
    x_coords = embedding_df.x_coord.tolist()
    y_coords = embedding_df.y_coord.tolist()
    
    source = ColumnDataSource(data=dict(x=x_coords, y=y_coords, names=words))
    p = figure(title="Task Specific Embeddings")
    p.scatter(x="x", y="y", size=8, source=source)
    labels = LabelSet(x="x", y="y", text="names", level="glyph", source=source, render_mode="canvas")
    p.add_layout(labels)
    show(p)


# Run plotting code
if __name__ == "__main__":
    model = load_model('../data/pos_tagger.h5')





