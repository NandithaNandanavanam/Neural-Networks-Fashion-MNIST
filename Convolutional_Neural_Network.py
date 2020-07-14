# Read Fashion MNIST dataset

import util_mnist_reader
X_train, y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

# Your code goes here . . .
#Convolutional Neural network
#-------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
                        
X_train = X_train / 255
X_test = X_test / 255

model = keras.Sequential()

model.add(keras.layers.Conv2D(64, kernel_size=2, activation="relu", padding = "same", input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(32, kernel_size=2, activation="relu", padding = "same"))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size = 10000, validation_data=(X_test, y_test), epochs = 10)

loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100)) 

#Compute confusion matrix
predictions = model.predict(X_test)
y_pred_labels = np.argmax(predictions, axis=1)
confusionmatrix = metrics.confusion_matrix(y_test, y_pred_labels)
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
