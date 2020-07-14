#Read Fashion MNIST dataset

import util_mnist_reader
X_train, Y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, Y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

# Your code goes here . . .
#One layer Neural network
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=51)

X_train = X_train/255.0
X_test = X_test/255.0

y_train = tf.keras.utils.to_categorical(Y_train, 10)
y_test = tf.keras.utils.to_categorical(Y_test, 10)
y_val = tf.keras.utils.to_categorical(Y_val, 10)

#Hyperparameters
m = 60000
input_nodes = 784     
hidden_nodes = 600
output_nodes = 10    
epochs = 60
learning_rate = 0.06

Loss_training = []
Loss_validation = []
Accuracy_training = []
Accuracy_validation =[]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(z):
    a = np.exp(z - z.max(axis=0, keepdims=True))
    return a / a.sum(axis = 0, keepdims=True)

def calculate_cost(A2, Y):
    cost = np.mean(-Y * (np.log(A2)))
    return cost

def model(X, Y, input_nodes, hidden_nodes, output_nodes, epochs, learning_rate):
    w_1 = np.random.randn(hidden_nodes, input_nodes)
    b_1 = np.ones((hidden_nodes, 1))
    w_2 = np.random.randn(output_nodes, hidden_nodes)
    b_2 = np.ones((output_nodes, 1))
        
    for epoch in range(0, epochs + 1):
        print("Epoch" , epoch)
        #Training
        z_1 = np.dot(w_1, X.T) + b_1
        a_1 = sigmoid(z_1)
        z_2 = np.dot(w_2, a_1) + b_2
        a_2 = softmax(z_2)
        Loss = calculate_cost(a_2,Y.T)        
        Loss_training.append(Loss)        
        
        y_pred_labels = np.argmax(a_2.T, axis=1)
        accuracy = metrics.accuracy_score(Y_train, y_pred_labels)     
        Accuracy_training.append(accuracy)
        
        d_z_2 = a_2 - Y.T
        d_w_2 = np.dot(d_z_2, a_1.T)/m
        d_b_2 = np.sum(d_z_2, axis=1, keepdims=True)/m
        
        w_2 = w_2 - learning_rate * d_w_2
        b_2 = b_2 - learning_rate * d_b_2
        
        d_z_1 = np.multiply(np.dot(w_2.T, d_z_2), 1-np.power(a_1, 2))
        d_w_1 = np.dot(d_z_1, X)/m
        d_b_1 = np.sum(d_z_1, axis=1, keepdims=True)/m
        
        w_1 = w_1 - learning_rate * d_w_1
        b_1 = b_1 - learning_rate * d_b_1
        
        #Validation
        z_1_val = np.dot(w_1 , X_val.T) + b_1
        a_1_val = sigmoid(z_1_val)
        z_2_val = np.dot(w_2, a_1_val) + b_2
        a_2_val = softmax(z_2_val)
        Loss_val = calculate_cost(a_2_val, y_val.T)
        Loss_validation.append(Loss_val)
        
        y_val_pred_labels = np.argmax(a_2_val.T, axis=1)
        val_accuracy = metrics.accuracy_score(Y_val, y_val_pred_labels)     
        Accuracy_validation.append(val_accuracy)
        
    return w_1,b_1,w_2,b_2

w_1,b_1,w_2,b_2 = model(X_train, y_train, input_nodes, hidden_nodes, output_nodes, epochs, learning_rate)

#Testing
z_1_test = np.dot(w_1, X_test.T) + b_1
a_1_test = sigmoid(z_1_test)
z_2_test = np.dot(w_2, a_1_test) + b_2
a_2_test = softmax(z_2_test)
test_cost = calculate_cost(a_2_test, y_test.T)

a_2_test_new = np.argmax(a_2_test.T, axis=1)

test_accuracy = metrics.accuracy_score(Y_test, a_2_test_new)
print("Test accuracy",test_accuracy)

#Compute Confusion matrix
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, a_2_test_new)
print(confusionmatrix)

#Graphs for Training errors and accuracy
fig = plt.figure(figsize=(10,10))
plt.subplots_adjust(left=None, bottom=0.1, right=1, top=0.9, wspace=0.5, hspace=0.2)
plt.subplot(221)
plt.plot(Loss_training, 'r--')
plt.xlabel("Number of epochs")
plt.ylabel("Training error")

plt.subplot(222)
plt.plot(Accuracy_training , 'g--')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")

plt.subplot(223)
plt.plot(Loss_validation , 'r--')
plt.xlabel("Number of epochs")
plt.ylabel("Validation Loss")

plt.subplot(224)
plt.plot(Accuracy_validation , 'g--')
plt.xlabel("Number of epochs")
plt.ylabel("Validation Accuracy")
