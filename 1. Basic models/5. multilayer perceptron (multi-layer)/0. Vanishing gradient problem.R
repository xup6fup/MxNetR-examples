library(mxnet)

#1. Call example data (iris)

data(iris)

X.array = array(as.matrix(iris[,-5]), dim = c(4, 150))
Y.array = array(t(model.matrix(~ -1 + iris[,5])), dim = c(3, 150))

#2. Define the data iterator

batch_size = 50

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
    idx = sample(dim(Y.array)[2], size = batch_size, replace = TRUE)
    data = mx.nd.array(X.array[,idx])
    label = mx.nd.array(Y.array[,idx])
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iter = my_iterator(batch_size)

#3. Define the model architecture (m-log loss function)

###Note:
###This is a 3-layer perceptron for iris data.
###The 1-layer perceptron model in '1. Basic models/4. multilayer perceptron (1-layer)'
###shows a good classification performance. However, if we add number of layer in our 
###model, it will face the vanishing gradient problem such as this example.

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 6, name = 'fc_layer_1')
sigmoid_layer_1 = mx.symbol.Activation(data = fc_layer_1, act.type = 'sigmoid', name = 'sigmoid_layer_1')
fc_layer_2 = mx.symbol.FullyConnected(data = sigmoid_layer_1, num.hidden = 8, name = 'fc_layer_2')
sigmoid_layer_2 = mx.symbol.Activation(data = fc_layer_2, act.type = 'sigmoid', name = 'sigmoid_layer_2')
fc_layer_3 = mx.symbol.FullyConnected(data = sigmoid_layer_2, num.hidden = 10, name = 'fc_layer_3')
sigmoid_layer_3 = mx.symbol.Activation(data = fc_layer_3, act.type = 'sigmoid', name = 'sigmoid_layer_3')
out_layer = mx.symbol.FullyConnected(data = sigmoid_layer_3, num.hidden = 3, name = 'out_layer')
softmax_layer = mx.symbol.SoftmaxOutput(data = out_layer, label = label, name = 'sofmax_layer')
m_logloss = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(softmax_layer), label))
loss_layer = mx.symbol.MakeLoss(m_logloss, name = 'm_logloss')

input_names = mxnet:::mx.model.check.arguments(m_logloss)

#4. Training model

#4-1. Build an executor to train model

my_executor = mx.simple.bind(symbol = loss_layer, data = c(4, batch_size), label = c(3, batch_size), ctx = mx.cpu(), grad.req = "write")

#4-2. Set the initial parameters

mx.set.seed(0)
new_arg = mxnet:::mx.model.init.params(symbol = loss_layer, input.shape = list(data = c(4, batch_size), label = c(3, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

#4-3. Define the optimizer

my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0, wd = 0, rescale.grad = 1, clip_gradient = 1)
my_updater = mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

#4-4. Start to train model

#Note: m-logloss will not be reduced.

my_iter$reset()

for (i in 1:3000) {
  
  my_iter$iter.next()
  
  ### Random input
  my_values <- my_iter$value()
  my_data <- my_values[input_names]
  mx.exec.update.arg.arrays(my_executor, arg.arrays = my_data, match.name = TRUE)
  mx.exec.forward(my_executor, is.train = TRUE)
  mx.exec.backward(my_executor)
  update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
  mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
  
  if (i %% 100 == 0) {
    cat(paste0("iteration = ", i, ": m-logloss = ", formatC(as.array(my_executor$ref.outputs$m_logloss_output), format = "f", 5), "\n"))
    Sys.sleep(0.1)
  }
  
}

#5. Visualize vanishing gradient problem

###Now let us re-train this model to see the gradient in each iteration

new_arg = mxnet:::mx.model.init.params(symbol = loss_layer, input.shape = list(data = c(4, batch_size), label = c(3, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

my_iter$reset()
mean_grad_list = NULL

for (i in 1:3000) {
  
  my_iter$iter.next()
  
  ### Random input
  my_values <- my_iter$value()
  my_data <- my_values[input_names]
  mx.exec.update.arg.arrays(my_executor, arg.arrays = my_data, match.name = TRUE)
  mx.exec.forward(my_executor, is.train = TRUE)
  mx.exec.backward(my_executor)
  update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
  mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
  
  mean_grad = sapply(my_executor$ref.grad.arrays[!names(my_executor$ref.grad.arrays)%in%input_names & !grepl('out', names(my_executor$ref.grad.arrays))], function (x) {mean(abs(as.array(x)))})
  mean_grad_list = rbind(mean_grad_list, mean_grad)
  
  if (i %% 100 == 0) {
    cat(paste0("iteration = ", i, ":\n"))
    cat(paste(paste(names(mean_grad), formatC(mean_grad, 4, format = 'f'), sep = ': '), collapse = '\n'), '\n')
    
    col_list = rainbow(ncol(mean_grad_list))
    
    for (j in 1:ncol(mean_grad_list)) {
      if (j == 1) {
        par(mar=c(4, 4, 0.5, 0.5))
        plot(1:i, mean_grad_list[,j], type = 'l', xlim = c(1, 3000), ylim = c(0, 0.1),
             xlab = 'iteration', ylab = 'mean(abs(gradients))', col = col_list[j])
      } else {
        lines(1:i, mean_grad_list[,j], col = col_list[j])
      }
    }
    legend('topright', colnames(mean_grad_list), col = col_list, lwd = 1, cex = 0.5)
    
    Sys.sleep(1)
  }
  
}

###We can find the vanishing gradient problem in first 3 layers.
###So how can we solve this problem?
