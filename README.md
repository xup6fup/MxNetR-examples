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

Now, you can follow these examples for futher learning MxNet. If you are a beginner, I suggest you can start from **1. Basic models**. Although these models are easy, and many other R-packages can achieve them. It may helps you to understand more clearly MxNet. All of examples in **1. Basic models** include **Standard example** and **Customize the iterator**. The **Standard example** is to use **mx.model.FeedForward.create** function to implement optimization, but this function only can handle limited data format and label format. The **Customize the iterator** step by step help you build your **loss function**, **data iterator**, and **optimizer**. This can help you to implement your specific models.

Tutorials
-------

[1. Basic models](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models)

[1-1. linear regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/1.%20linear%20regression)

[1-2. logistic regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/2.%20logistic%20regression)

[1-3. softmax regression](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/3.%20softmax%20regression)

[1-4. multilayer perceptron (1-layer)](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/4.%20multilayer%20perceptron%20(1-layer))

[1-5. multilayer perceptron (multi-layer)](https://github.com/xup6fup/MxNetR-examples/tree/master/1.%20Basic%20models/5.%20multilayer%20perceptron%20(multi-layer))

Tutorial 1-5 includes a **vanishing gradient problem**, and this example can help users to solve this problem.

[2. Convolutional Neural Networks](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks)

[2-1. LeNet for MNIST](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks/1.%20LeNet%20for%20MNIST)

[2-2. Pre-trained model usage](https://github.com/xup6fup/MxNetR-examples/tree/master/2.%20Convolutional%20Neural%20Networks/2.%20Pre-trained%20model%20usage)

License
-------
MXNet R-package is licensed under [Apache-2.0](./LICENSE) license.
