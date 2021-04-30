import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D as MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

mod=Sequential()
mod.add(Conv2D(32, (3,3), padding='same',input_shape=(28,28,1)))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Conv2D(32,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Conv2D(32,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2, 2)))
#mod.add(Conv2D(32,(3,3),activation='relu'))
#mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Flatten())
mod.add(Dense(units = 128, activation = 'relu'))
mod.add(Dense(units=10,activation="softmax"))


mod.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'] )
mod.fit(train_ind,train_dep,epochs=1)

test_inde= test_ind.reshape(test_ind.shape[0], 28,28, 1)
test_ind.shape
test_dep=to_categorical(test_dep)
test_dep.shape
test_loss, test_acc = mod.evaluate(test_ind, test_dep)
print('Test accuracy:', test_acc)

predictions = mod.predict(test_inde[0])
#test_inde[0].shape
predictions[0]

np.argmax(predictions[0])

test_labels[0]