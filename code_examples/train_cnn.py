# Let's load the data we'll use
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print "Training data:", X_train.shape
print "Training labels:", y_train.shape

# Input tensor is just a collection of 3D tensors
# Each is an image with 3 color channels
X_train = np.array([img / 255. for img in X_train]) # Min-max normalization

# Imports for the model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten
from keras.layers import Conv2D, Dense, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

model.fit(X_train[:25000], y_train[:25000], epochs=10)
model.save('../data/cifar_cnn.h5')