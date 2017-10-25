# MxNetR-examples

<img src="https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnetR.png" width="160" height="60" alt="MxNetR"/><img src="https://assets-cdn.github.com/images/topics/r.png?v=1508892931" width="60" height="60" alt="RLogo"/>

This project provides a series of example for letting readers to get started quickly. Especially, these examples were all based on R langugue, and the whole file is a Rstudio project. 
First, you need to instal the MXNet package in your computer ([here](https://mxnet.incubator.apache.org/get_started/windows_setup.html#install-mxnet-for-r) is the official website). MXNet for R is available for both CPUs and GPUs. For Windows users, MXNet provides prebuilt binary packages. You can install the package directly in the R console.

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

To use the GPU version or to use it on Linux, please follow [Installation Guide](http://mxnet.io/get_started/install.html)

Now, you can follow these examples for futher learning MxNet. If you are a beginner, I suggest you can start from **1. Basic models**. Although these models are easy, and many other R-packages can achieve them. It may helps you to understand more clearly MxNet. All of examples in **1. Basic models** include **Standard example** and **Customize the iterator**. The **Standard example** is to use **mx.model.FeedForward.create** function to implement optimization, but this function only can handle limited data format and label format. The **Customize the iterator** step by step help you build your **loss function**, **data iterator**, and **optimizer**. This can help you to implement your special models.

Tutorials
-------

**[1. Basic models](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models)**

Regression and classification are classical task of machine learning. They are easy to be understanded. Following simple examples will help you step by step to learn the principle of MxNet.

[1-1. linear regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/1.%20linear%20regression)

Linear regression is the best entry example for beginners. This tutorial can let you know the Symbol API in MxNet. Let us stack symbols to complete a linear regression (you can think of it as a 0-layer neural network).

[1-2. logistic regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/2.%20logistic%20regression)

Logistic regression is very similar with linear regression, and the only difference between them is output form. For this difference, we need to modify the loss function for training this model.

[1-3. softmax regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/3.%20softmax%20regression)

Softmax regression is also similar with linear regression and logistic regression. It is worth mentioning that we may need to change the format of label as one hot encoding array.

[1-4. multilayer perceptron (1-layer)](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/4.%20multilayer%20perceptron%20(1-layer))

Under MxNet environment, the difference between multilayer perceptron and regression is to add only few layers. Before we start to build a more complex model, we need to pay attention the output shape at all times. The function **'mx.symbol.infer.shape'** can help you to do this work, please learn to use it skillfully.

[1-5. multilayer perceptron (multi-layer)](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/5.%20multilayer%20perceptron%20(multi-layer))

Deep learning is so easy using MxNet? Although it looks just adding few layers but not yet! **Vanishing gradient problem** is a major issue in deep learning training. This tutorial includes a **vanishing gradient problem**, and this example can help users to visualize and solve this problem.

**[2. Convolutional Neural Networks](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks)**

Convolutional Neural Network (CNN) is a class of deep, feed-forward artificial neural networks that has successfully been applied to analyzing visual imagery. Moreover, it also has been applied in some natural language processing tasks.

[2-1. LeNet for MNIST](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks/1.%20LeNet%20for%20MNIST)

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. This example can help users fastly learning to use convolutional neural networks for analyzing image.

[2-2. Pre-trained model usage](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks/2.%20Pre-trained%20model%20usage)

Training a complex convolutional neural networks will take so much time! This example can let users to know how to use pre-trained model.

License
-------
MXNet R-package is licensed under [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/R-package/LICENSE) license.
