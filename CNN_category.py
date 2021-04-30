import tensorflow as tf
import keras 
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

import matplotlib.pyplot as plt

data = fashion_mnist
(train_ind, train_dep),(test_ind, test_dep) = data.load_data()
'''print(train_ind.shape)
print(train_dep.shape)
print(test_ind.shape)
print(test_dep.shape)
print(train_ind[0])  # pixel of 1 row of indepent
print(train_dep[0])  # num 5
'''
plt.imshow(train_ind[0]) # image of particular pixel

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Ankleboots', 'Bag']

train_ind = train_ind.astype('float32')
test_ind = test_ind.astype('float32')

train_ind /= 255
test_ind  /= 255

train_ind = train_ind.reshape(train_ind.shape[0], 28, 28, 1)
train_ind.shape
train_dep = to_categorical(train_dep)

m= Sequential()
m.add(Conv2D(32, (3), padding = 'same', input_shape = (28, 28, 1), activation = 'relu'))
m.add(MaxPooling2D(pool_size= (2,2)))
m.add(Conv2D(32, (3), activation = 'relu'))
m.add(MaxPooling2D(pool_size= (2,2)))
m.add(Conv2D(32, (3), activation = 'relu'))
m.add(MaxPooling2D(pool_size= (2,2)))
m.add(Flatten())
m.add(Dense(units = 128, activation= 'relu'))
m.add(Dense(units = 10, activation= 'softmax'))

m.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
m.fit(train_ind, train_dep, epochs = 1)

test_ind = test_ind.reshape(test_ind.shape[0], 28,28,1)
test_dep =to_categorical(test_dep)
test_loss, test_acc= m.evaluate(test_ind, test_dep)

'''
test_ind = test_ind.reshape(-1,28,28,1)
test_dep =to_categorical(test_dep)
test_loss, test_acc= m.evaluate(test_ind, test_dep)
'''

print(train_ind[0].shape)
print(test_ind[0].shape)

print('Test Accuracy: ',test_acc)
test_dep[0]
predictions = m.predict(test_dep[0])
#test_ind[0].shape
predictions

#np.argmax(predictions[0])




































