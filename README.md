# MxNetR-examples

![R](https://assets-cdn.github.com/images/topics/r.png?v=1508892931)
![MxNetR](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnetR.png)

This project provides a series of example for letting readers to get started quickly. Especially, these examples were all based on R langugue, and the whole file is a Rstudio project. 
First, you need to instal the MXNet package in your computer ([here](https://mxnet.incubator.apache.org/get_started/windows_setup.html#install-mxnet-for-r) is the official website). MXNet for R is available for both CPUs and GPUs. For Windows users, MXNet provides prebuilt binary packages. You can install the package directly in the R console.

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

Now, you can follow these examples for futher learning MxNet.

To use the GPU version or to use it on Linux, please follow [Installation Guide](http://mxnet.io/get_started/install.html)

License
-------
MXNet R-package is licensed under [Apache-2.0](./LICENSE) license.