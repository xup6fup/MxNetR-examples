library(mxnet)

#1. Generate example data

set.seed(0)

X1 = rnorm(1000) 
X2 = rnorm(1000) 
Y = X1 * 0.7 + X2 * 1.3 - 3.1 + rnorm(1000)

X.array = array(rbind(X1, X2), dim = c(2, 1000))
Y.array = array(Y, dim = c(1, 1000))

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
    idx = sample(length(Y.array), size = batch_size, replace = TRUE)
    data = mx.nd.array(X.array[,idx])
    label = mx.nd.array(Y.array[,idx])
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iter = my_iterator(batch_size)

#3. Define the model architecture (RMSE loss function)

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer = mx.symbol.FullyConnected(data = data, num.hidden = 1, name = 'fc_layer')
out_layer = mx.symbol.MakeLoss(mx.symbol.mean(mx.symbol.square(mx.symbol.Reshape(fc_layer, shape = 0) - label)), name = 'rmse')

input_names = mxnet:::mx.model.check.arguments(out_layer)

#4. Training model

#4-1. Build an executor to train model

my_executor = mx.simple.bind(symbol = out_layer, data = c(2, batch_size), ctx = mx.cpu(), grad.req = "write")

#4-2. Set the initial parameters

mx.set.seed(0)
new_arg = mxnet:::mx.model.init.params(symbol = out_layer, input.shape = list(data = c(2, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

#4-3. Define the optimizer

my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0, wd = 0, rescale.grad = 1, clip_gradient = 1)
my_updater = mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

#4-4. Start to train model

my_iter$reset()

for (i in 1:5000) {
  
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
    cat(paste0("iteration = ", i, ": Root-Mean-Square Error (RMSE) = ", formatC(as.array(my_executor$ref.outputs$rmse_output), format = "f", 5), "; weight = ", paste(formatC(as.array(my_executor$ref.arg.arrays$fc_layer_weight), format = "f", 4), collapse = ", "), "; bias = ", formatC(as.array(my_executor$ref.arg.arrays$fc_layer_bias), format = "f", 4), "\n"))
    Sys.sleep(0.1)
  }
  
}

#5. Save and load model

#5-1. Save model (Note: we need to save the 'fc_layer' but not the 'out_layer')

mx.symbol.save(fc_layer, filename = "model/linear_regression.json")
mx.nd.save(my_executor$arg.arrays[!names(my_executor$arg.arrays)%in%input_names], filename = "model/linear_regression_arg_params.params")
mx.nd.save(my_executor$aux.arrays, filename = "model/linear_regression_aux_params.params")

#5-2. Load model

My_sym = mx.symbol.load("model/linear_regression.json")
My_arg_params = mx.nd.load("model/linear_regression_arg_params.params")
My_aux_params = mx.nd.load("model/linear_regression_aux_params.params")

#5-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data = c(2, 1000), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data = mx.nd.array(X.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

predict_Y = My_exec$ref.outputs$fc_layer_output
RMSE = mean((as.array(predict_Y) - Y.array)^2)
print(RMSE)
