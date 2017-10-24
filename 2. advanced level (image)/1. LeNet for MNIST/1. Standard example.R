library(mxnet)

#1. Generate example data

#1-1. Download MNIST data

load_image_file <- function(conn) {
  readBin(conn, 'integer', n=1, size=4, endian='big')
  n = readBin(conn, 'integer', n=1, size=4, endian='big')
  nrow = readBin(conn, 'integer', n=1, size=4, endian='big')
  ncol = readBin(conn, 'integer', n=1, size=4, endian='big')
  x = readBin(conn, 'integer', n=n*nrow*ncol, size=1, signed=F)
  x = matrix(x,  ncol=nrow * ncol,  byrow=T)
  close(conn)
  return(x)
}

load_label_file <- function(conn) {
  readBin(conn, 'integer', n=1, size=4, endian='big')
  n = readBin(conn, 'integer', n=1, size=4, endian='big')
  y = readBin(conn, 'integer', n = n, size=1, signed=F)
  close(conn)
  return(y)
}

download.mnist <- function(range = c(0, 1), global = FALSE) {
  mnist <- list(
    train = list(),
    test = list()
  )
  
  mnist$train$x <- load_image_file(gzcon(url("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "rb")))
  mnist$test$x <- load_image_file(gzcon(url("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "rb")))
  mnist$train$y <- load_label_file(gzcon(url("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "rb")))
  mnist$test$y <- load_label_file(gzcon(url("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "rb")))
  
  # Normalize the x's - only if needed
  if (!isTRUE(all.equal(c(0, 255), range))) {
    if (! is.numeric(range))
      stop("'range' must be a numeric vector")
    if (length(range) != 2)
      range <- range(range)
    
    mnist$train$x <-  mnist$train$x / (255 / diff(range)) + range[1]
    mnist$test$x <-  mnist$test$x / (255 / diff(range)) + range[1]
    
    # Convert to integer if possible
    s <- seq(range[1], range[2], length.out = 256)
    if (isTRUE(all.equal(s, as.integer(s)))) {
      storage.mode(mnist$train$x) <- "integer"
      storage.mode(mnist$test$x) <- "integer"
    }
  }
  
  if (global) {
    save(mnist, file = paste(system.file(package="mnist"), "data", "mnist.RData", sep=.Platform$file.sep))
    assign("mnist", mnist, envir = globalenv())
  }
  
  return(mnist)
}

mnist_data = download.mnist()

#1-2. Reshape data

Train.X.array = array(t(mnist_data$train$x), dim = c(28, 28, 1, 60000))
Train.Y = mnist_data$train$y

Test.X.array = array(t(mnist_data$test$x), dim = c(28, 28, 1, 10000))
Test.Y = mnist_data$test$y


#2. Define the model architecture

data <- mx.symbol.Variable('data')

conv1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
relu1 = mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 = mx.symbol.Pooling(data = relu1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

conv2 = mx.symbol.Convolution(data = pool1, kernel = c(5, 5), num_filter = 50)
relu2 = mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 = mx.symbol.Pooling(data = relu2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

flatten = mx.symbol.Flatten(data = pool2)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden=500)
relu3 = mx.symbol.Activation(data = fc1, act_type = "relu")

fc2 = mx.symbol.FullyConnected(data = relu3, num_hidden = 10)
lenet = mx.symbol.SoftmaxOutput(data = fc2)

#3. Training model (about 10-15 mins)

n.cpu <- 4
device.cpu <- lapply(0:(n.cpu-1), function(i) {mx.cpu(i)})

logger = mx.metric.logger$new()

lenet_model = mx.model.FeedForward.create(lenet, X = Train.X.array, y = Train.Y,
                                          ctx = device.cpu, num.round = 20, array.batch.size = 100,
                                          learning.rate = 0.05, momentum = 0.9, wd = 0.00001,
                                          eval.metric = mx.metric.accuracy,
                                          epoch.end.callback = mx.callback.log.train.metric(5, logger))

#4. Save and load model

#4-1. Save model

mx.model.save(lenet_model, "model/lenet_mnist", iteration = 0)

#4-2. Load model

My_model = mx.model.load("model/lenet_mnist", iteration = 0)

#4-3. Inference

preds = predict(My_model, Test.X.array)
pred.label = max.col(t(preds)) - 1
tab = table(pred.label, Test.Y)
cat("Testing accuracy rate =", sum(diag(tab))/sum(tab))
print(tab)