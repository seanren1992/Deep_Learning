# Deep Learning


<br/>

## Machine Learning 

- **`Supervised learning`**: Supervised learning algorithms are a class of machine learning algorithms that use previously-labeled data to learn its features, so they can classify similar but unlabeled data.
- **`Linear and logistic regression`**: Using features of the input data to predict a value
- **`Support vector machines`**: SVM is a supervised machine learning algorithm that is used for classification. 
- **`Decision Trees`**: A decision tree creates a classifier in the form of a tree.
- **`Naive Bayes`**: Naive Bayes is different from many other machine learning algorithms.
- **`Unsupervised learning`**: Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets      consisting of input data without labeled responses.
- **`K-means`**: K-means is a clustering algorithm that groups the elements of a dataset into k distinct clusters.
- **`Reinforcement learning`**: Reinforcement is about taking suitable action to maximize reward in a particular situation. 
- **`Q-learning`**: Q-learning is an off-policy temporal-difference reinforcement learning algorithm. 
<br/>


## Neural Networks

**A neuron is a mathematical function that takes one or more input values, and outputs a single numerical value**
![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/Neurons.JPG)

**A neural network can have an indefinite number of neurons, which are organized in interconnected layers.** 
![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/Layers.JPG)

**The following diagram demonstrates a 3-layer fully connected neural network with two hidden layers.** 
![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/Multi-layer.JPG)

**The neurons and their connections form directed cyclic graphs. In such a graph, the information cannot pass twice from the same neuron (no loops) and it flows in only one direction, from the input to the output.**
![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/Directed%20Cyclic%20Graphs.JPG)


## Deep Learning 

![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/Deep%20Learning.JPG)

#### Open Source Libraries
- TensorFlow
- Keras
- PyTorch

Using Keras to classify handwritten digits： 
 - **download the datasets using Keras:**
```python
from keras.datasets import mnist
```
- **Importing a few classes to use a feed-forward network:**
```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
```
- **Training and testing data:**
```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```
- **Modifying the data to be able to use it:**
```python
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
```
- **The labels indicate the value of the digit depicted in the images;**
```python
classes = 10
Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)
```
- **Setting the size of the input layer (the size of the MNIST images), the number of hidden neurons, the number of epochs to train the network, and the mini batch size:** 
```python
input_size = 784
batch_size = 100
hidden_neurons = 100
epochs = 100
```
- **Applying the Sequential model:** 
```python
model = Sequential([
    Dense(hidden_neurons, input_dim=input_size),
    Activation('sigmoid'),
    Dense(classes),
    Activation('softmax')
])
```
- **Cross-entropy and stochastic gradient descent:**
```python
model.compile(loss='categorical_crossentropy',
metrics=['accuracy'], optimizer='sgd')
```
<br/> 

## Computer Vision 

**Convolutional layers**
The convolutional layer is the most important building block of a CNN. It consists of a set of filters (also known as kernels or feature detectors), where each filter is applied across all areas of the input data. A filter is defined by a set of learnable weights. 

![alt text](https://github.com/David-SF2290/Deep-Learning/blob/master/Graph_Doc/CNN.JPG)



## Recurrent Neural Networks





## Reinforcement Learning





## Deep Reinforcement Learning



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

