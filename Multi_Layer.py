# Read Fashion MNIST dataset

import util_mnist_reader
X_train, y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

# Your code goes here . . .
#Multi layer Neural network
#-------------------------------------------------------------------------------------------------------------------------------
#libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

(X_val, X_test, Y_val, Y_test) = train_test_split(X_test, y_test, test_size=0.50 , random_state = 2)

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (784,)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(30, activation='sigmoid'))
model.add(keras.layers.Dense(20, activation='sigmoid'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, Y_val), epochs=40, batch_size=1000)

loss, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Compute confusion matrix
predictions = model.predict(X_test)
y_pred_labels = np.argmax(predictions, axis=1)
confusionmatrix = metrics.confusion_matrix(Y_test, y_pred_labels)
print(confusionmatrix)

#Graphs for Training errors and accuracy
fig = plt.figure(figsize=(10,10))
plt.subplots_adjust(left=None, bottom=0.1, right=1, top=0.9, wspace=0.5, hspace=0.2)
plt.subplot(221)
plt.plot(history.history['loss'],'r--')
plt.xlabel("Number of epochs")
plt.ylabel("Training error")

plt.subplot(222)
plt.plot(history.history['acc'],'g--')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")

plt.subplot(223)
plt.plot(history.history['val_loss'],'r--')
plt.xlabel("Number of epochs")
plt.ylabel("Validation Loss")

plt.subplot(224)
plt.plot(history.history['val_acc'],'g--')
plt.xlabel("Number of epochs")
plt.ylabel("Validation Accuracy")
