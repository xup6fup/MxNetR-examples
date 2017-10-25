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

Now, you can follow these examples for futher learning MxNet. If you are a beginner, I suggest you can start from '1. Basic models'. Although these models are easy, and many other R-packages can achieve them. It may helps you to understand more clearly MxNet. All of examples in '1. Basic models' include 'Standard example' and 'Customize the iterator'. The 'Standard example' is to use 'mx.model.FeedForward.create' function to implement optimization, but this function only can handle limited data format and label format. The 'Customize the iterator' step by step help you build your **loss function**, **data iterator**, and **optimizer**. This can help you to implement your specific models.


License
-------
MXNet R-package is licensed under [Apache-2.0](./LICENSE) license.
