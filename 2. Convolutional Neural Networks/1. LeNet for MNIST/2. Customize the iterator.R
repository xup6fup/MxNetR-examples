library(mxnet)
library(imager)

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
Train.Y.array =  array(t(model.matrix(~ -1 + as.factor(mnist_data$train$y))), dim = c(10, 60000))

Test.X.array = array(t(mnist_data$test$x), dim = c(28, 28, 1, 10000))
Test.Y = mnist_data$test$y

#1-3. Display image data

par(mar = rep(0, 4), mfcol = c(5, 5))
for (i in 1:25) {
  plot(NA, xlim = 0:1, ylim = 0:1, xaxt = "n", yaxt = "n", bty = "n")
  img = as.raster(t(matrix(as.numeric(Train.X.array[,,,i]), nrow = 28)))
  rasterImage(img, -0.04, -0.04, 1.04, 1.04, interpolate=FALSE)
  text(0.05, 0.95, mnist_data$train$y[i], col = "green", cex = 1.5)
}

#2. Define the data iterator

batch_size = 100

my_iterator = function(batch_size) {
  
  batch = 0
  batch_per_epoch = 5
  
  reset = function() {batch <<- 0}
  
  iter.next = function() {
    batch <<- batch+1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
  }
  
  value = function() {
    set.seed(123+batch)
    idx = sample(dim(Train.Y.array)[2], size = batch_size, replace = TRUE)
    data = mx.nd.array(array(Train.X.array[,,,idx], dim = c(28, 28, 1, batch_size)))
    label = mx.nd.array(Train.Y.array[,idx])
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iter = my_iterator(batch_size)

#3. Define the model architecture (m-log loss function)

data = mx.symbol.Variable('data')
label = mx.symbol.Variable(name = 'label')

conv1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20, name = 'conv1')
relu1 = mx.symbol.Activation(data = conv1, act_type = "relu", name = 'relu1')
pool1 = mx.symbol.Pooling(data = relu1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2), name = 'pool1')

conv2 = mx.symbol.Convolution(data = pool1, kernel = c(5, 5), num_filter = 50, name = 'conv2')
relu2 = mx.symbol.Activation(data = conv2, act_type = "relu", name = 'relu1')
pool2 = mx.symbol.Pooling(data = relu2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2), name = 'pool2')

flatten = mx.symbol.Flatten(data = pool2)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500, name = 'fc1')
relu3 = mx.symbol.Activation(data = fc1, act_type = "relu", name = 'relu3')

fc2 = mx.symbol.FullyConnected(data = relu3, num_hidden = 10, name = 'fc2')
lenet = mx.symbol.SoftmaxOutput(data = fc2, label = label, name = 'lenet')

m_logloss = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(lenet), label))
out_layer = mx.symbol.MakeLoss(m_logloss, name = 'm_logloss')

input_names = mxnet:::mx.model.check.arguments(out_layer)

#4. Training model

#4-1. Build an executor to train model

my_executor = mx.simple.bind(symbol = out_layer, data = c(28, 28, 1, batch_size), label = c(10, batch_size), ctx = mx.cpu(), grad.req = "write")

#4-2. Set the initial parameters

mx.set.seed(0)
new_arg = mxnet:::mx.model.init.params(symbol = out_layer, input.shape = list(data = c(28, 28, 1, batch_size), label = c(10, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

#4-3. Define the optimizer

my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0, wd = 0, rescale.grad = 1, clip_gradient = 1)
my_updater = mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

#4-4. Start to train model

my_iter$reset()
mean_grad_list = NULL

for (i in 1:10000) {
  
  my_iter$iter.next()
  
  ### Random input
  my_values <- my_iter$value()
  my_data <- my_values[input_names]
  mx.exec.update.arg.arrays(my_executor, arg.arrays = my_data, match.name = TRUE)
  mx.exec.forward(my_executor, is.train = TRUE)
  mx.exec.backward(my_executor)
  update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
  mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
  
  mean_grad = sapply(my_executor$ref.grad.arrays[c(2, 4, 6, 8)], function (x) {mean(abs(as.array(x)))})
  mean_grad_list = rbind(mean_grad_list, mean_grad)
  
  if (i %% 500 == 0) {
    cat(paste0("iteration = ", i, ": m-logloss = ", formatC(as.array(my_executor$ref.outputs$m_logloss_output), format = "f", 5), "\n"))
    
    col_list = rainbow(ncol(mean_grad_list))
    
    for (j in 1:ncol(mean_grad_list)) {
      if (j == 1) {
        par(mar=c(4, 4, 0.5, 0.5), mfcol = c(1, 1))
        plot(1:i, mean_grad_list[,j], type = 'l', xlim = c(1, 10000), ylim = c(0, 1),
             xlab = 'iteration', ylab = 'mean(abs(gradients))', col = col_list[j], lwd = 0.1)
      } else {
        lines(1:i, mean_grad_list[,j], col = col_list[j], lwd = 0.1)
      }
    }
    legend('topright', colnames(mean_grad_list), col = col_list, lwd = 1, cex = 0.7)
    
    Sys.sleep(0.1)
  }
  
}

#5. Save and load model

#5-1. Save model (Note: we need to save the 'softmax_layer' but not the 'out_layer')

mx.symbol.save(lenet, filename = "model/lenet.json") 
mx.nd.save(my_executor$arg.arrays[!names(my_executor$arg.arrays)%in%input_names], filename = "model/lenet_arg_params.params")
mx.nd.save(my_executor$aux.arrays, filename = "model/lenet_aux_params.params")

#5-2. Load model

My_sym = mx.symbol.load("model/lenet.json")
My_arg_params = mx.nd.load("model/lenet_arg_params.params")
My_aux_params = mx.nd.load("model/lenet_aux_params.params")

#5-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data = c(28, 28, 1, 10000), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data = mx.nd.array(Test.X.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

preds = as.array(My_exec$ref.outputs$lenet_output)
pred.label = max.col(t(preds)) - 1
tab = table(pred.label, Test.Y)
cat("Testing accuracy rate =", sum(diag(tab))/sum(tab))
print(tab)
