# Fashion MNIST Classification using Neural Network 
Classify apparel images in Fashion-MNIST dataset using custom built fully-connected neural network

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_intro.png?raw=true" width="800" height="400">

## Features
⚡Multi Label Image Classification  
⚡Cutsom Fully Connected NN  
⚡Fashion MNIST  
⚡PyTorch

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [License](#license)
- [Get in touch](#get-in-touch)

## Introduction
Just like [MNIST digit classification](https://github.com/sssingh/hand-written-digit-classification), the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a popular dataset for classification in the Machine Learning community for building and testing neural networks. MNIST is a pretty trivial dataset to be used with neural networks where one can quickly achieve better than 97% accuracy. Experts recommend ([Ian Goodfellow](https://twitter.com/goodfellow_ian/status/852591106655043584), [François Chollet](https://twitter.com/fchollet/status/852594987527045120)) to move away from MNIST dataset for model benchmarking and validation. Fashion-MNIST is more complex than MNIST, and it's a much better dataset for evaluating models than MNIST.

## Objective
We'll build a neural network using PyTorch. Only `fully-connected` layers will be used. The goal here is to classify ten classes of apparel images in the Fashion-MNIST dataset with as high accuracy as possible by only using fully-connected layers (i.e., without using `Convolution` layers)

## Dataset
- Dataset consists of 60,000 training images and 10,000 testing images.
- Every image in the dataset will belong to one of the ten classes...

| Label	| Description |
|--- | ---|
|0|	T-shirt/top|
|1|	Trouser|
|2|	Pullover|
|3|	Dress|
|4|	Coat|
|5|	Sandal|
|6|	Shirt|
|7|	Sneaker|
|8|	Bag|
|9|	Ankle boot|

- Each image in the dataset is a 28x28 pixel grayscale image, a zoomed-in single image shown below...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_single_image.png?raw=true">


- Here are zoomed-out samples of other images from the training dataset with their respective labels...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_samples.png?raw=true">


- We will use the in-built Fashion-MNIST dataset from PyTorch's `torchvision` package. The advantage of using the dataset this way is that we get a clean pre-processed dataset that pairs the image and respective label nicely, making our life easier when we iterate through the image samples while training and testing the model. Alternatively, the raw dataset can be downloaded from the original source. Like MNIST, the raw dataset comes as a set of 4 zip files containing training images, training image labels, testing images, and testing image labels in separate files... 
[train-images](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz), 
[train-labels](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz), 
[test-images](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz), 
[test-labels](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)

## Evaluation Criteria

### Loss Function  
Negative Log-Likelihood Loss (NLLLoss) is used as the loss function during model training and validation 

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/logsoftmax.png?raw=true">

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/nllloss.png?raw=true">

<br>Note the `negative` sign in front `NLLLoss` formula (and in the `BCELoss` formula as well) hence negative in the name. The negative sign is put in front to make the average loss positive. Suppose we don't do this since the `log` of a number less than 1 is negative. In that case, we will have a negative overall average loss. To reduce the loss, we need to `maximize` the loss function instead of `minimizing,` which is a much easier task mathematically than `maximizing.`

### Performance Metric
`accuracy` is used as the model's performance metric on the test-set 

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/accuracy.png?raw=true">

## Solution Approach
- Training dataset of 60,000 images and labels along with testing dataset of 10,000 images and labels are downloaded from torchvision.
- Training dataset is then further split into training (80% i.e. 48,000 samples) and validation (20% i.e. 12,000) samples sets
- The training, validation, and testing datasets are then wrapped in PyTorch `DataLoaders` objects so that we can iterate through them with ease. Again, a `batch_size` of 64 is used.
- The neural network is implemented using the `Sequential` wrapper class from PyTorch `nn` module. 
- We start with precisely the same simple network we used with the MNIST dataset; the network is defined to have... 
    - The implicit input layer has a size of 784 (28x28 flattened)
    - The first hidden layer has 784 input and produces 128 `ReLU` activated outputs
    - A dropout layer with a 25% probability
    - The second hidden layer has 128 input and produces 64 `ReLU` activated outputs
    - A dropout layer with a 25% probability
    - Output layer has 64 inputs and produces ten logits outputs Corresponding to each of the ten classes in the dataset). Logits are then passed through `LogSoftmax` activation to get `log-probability` for each class. 

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_model1.png?raw=true">
    
- Network is then trained for 25 epochs using the `NLLLoss` loss function and `Adam` optimizer with a learning rate of `0.003`.
- We keep track of training loss and plot it; we observe training loss decreasing throughout, but validation loss does not improve further after 11-12 epochs...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_loss1.png?raw=true">

- We have a trained model; let's try classifying a single image from the test set. 

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/predict1.png?raw=true">

We observe that the network can identify the image with a high degree of accuracy. This looks good; let's evaluate our trained network on a complete test set. We managed to get a prediction `accuracy` of `87.42%`, which is not bad given that our model is a simple neural net.

To improve prediction accuracy, we add a few more fully-connected layers with dropouts. Modified model architecture is shown below...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_model2.png?raw=true">

- We then re-train the new model for 35 epochs using the `NLLLoss` loss function and `Adam` optimizer with a learning rate of `0.0007`. 
- We keep track of training and validation losses and plot them. We observe that the modified model can produce lower validation loss compared to the previous model

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_loss2.png?raw=true">

- We then evaluate our trained network to complete the test set. This time we managed to get a prediction `accuracy` of `88.65%`, which is more than a percent improvement over the previous model.
- It's possible to improve model performance and push it above 90% by further experimenting with the architecture and tuning hyperparameters. However, it'd be tough to get the accuracy in the range of 95-98% using a model just utilizing fully connected layers. This is because Fashion-MNIST is a more complex dataset compared to MNIST. For example, when images of 28x28 are flattened to make them a vector of 784 elements to feed to a fully connected layer, they lose `spatial` structural information; hence model's ability to learn the underlying structure is reduced.
Instead of fully-connected layers, if we were to use `Convolution` layers that could consume 28x28 images directly preserving the `spatial` structural information, it'd be pretty easy to obtain accuracy above 95%.

## How To Use
1. Ensure the below-listed packages are installed
    - `NumPy`
    - `matplotlib`
    - `torch`
    - `torchvision`
2. Download `fashion_mnist_classification_nn_pytorch.ipynb` jupyter notebook from this repo
3. Execute the notebook from start to finish in one go. If a GPU is available (recommended), it'll use it automatically; otherwise, it'll fall back to the CPU. 
4. A machine with `NVIDIA Quadro P5000` GPU with 16GB memory takes approximately 15 minutes to train and validate for 35 epochs.
5. A trained model can be used to predict, as shown below...

```python
    # Flatten the image to make it 784 1D tensor
    image = image.view(image.shape[0], -1)
    # Predict and get probabilities
    proba = torch.exp(model(image))
    # Extract the most likely predicted class
    _, pred_label = torch.max(proba, dim=1)
    print(pred_label)
```

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

# Contact Me
[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sunil@sunilssingh.me)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/@thesssingh)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sssingh/)
[![website](https://img.shields.io/badge/web_site-8B5BE8?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sunilssingh.me)
[Back To The Top](#Fashion-MNIST-Classification-using-Neural-Network)
