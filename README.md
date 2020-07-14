# Neural-Networks-Fashion-MNIST
## Dataset
The Fashion-MNIST dataset consists of 60000 training and 10000 testing images which are a dataset of Zalando’s article images. Each image is of size 28 x 28 and is in grayscale. The pixel values are between 0 to 255. Each image is assigned to one of the 10 labels. The 10 labels are namely: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle Boot.
## Training
The Fashion-MNIST dataset contains 60000 training images and 10000 testing images.  The model parameters: weights of features and bias are initialized. The hyper parameters: learning rate, number of hidden nodes in the hidden layer and epochs are set to a value initially. One-hot-encode is applied on the output vector that creates a column for each output category and a binary variable is inputted for each category.
## Result
### Confusion Matrix

A single layer neural network model with 600 hidden nodes and 60 iterations and a learning rate of 0.6 hardly yielded 60.3% accuracy. Also, the computation time was very slow.  
 
Multi-layer neural network model with 100, 30, 20, 10 nodes in respective hidden layers after an iteration of 40 yielded better accuracy of 88.42% as well as lesser computation time. 
 
Convolutional neural network model was observed to give the best accuracy at the least time as even after an iteration of 3, it was able to achieve an accuracy of 71% unlike single and multi-layer neural network models that took more computations to reach this level. At 10 iterations, the accuracy increased to 81.96%.

