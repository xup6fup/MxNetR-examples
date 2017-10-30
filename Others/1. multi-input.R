library(mxnet)

#1. Generate example data

set.seed(0)

X1 = rnorm(1000) 
X2 = rnorm(1000)
X3 = rnorm(1000) 
X4 = rnorm(1000) 
Y = X1 * 0.7 + X2 * 1.3 - X3 * 0.9 - X4 * 1.7 - 3.1 + rnorm(1000)

X1.array = array(rbind(X1, X2), dim = c(2, 1000))
X2.array = array(rbind(X3, X4), dim = c(2, 1000))
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
    data1 = mx.nd.array(X1.array[,idx])
    data2 = mx.nd.array(X2.array[,idx])
    label = mx.nd.array(Y.array[,idx])
    return(list(data1 = data1, data2 = data2, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iter = my_iterator(batch_size)

#3. Define the model architecture (RMSE loss function)

data1 = mx.symbol.Variable(name = 'data1')
data2 = mx.symbol.Variable(name = 'data2')
label = mx.symbol.Variable(name = 'label')

fc1_layer = mx.symbol.FullyConnected(data = data1, num.hidden = 1, name = 'fc1_layer')
tanh1_layer = mx.symbol.Activation(data = fc1_layer, act.type = 'tanh')
fc2_layer = mx.symbol.FullyConnected(data = data2, num.hidden = 1, name = 'fc2_layer')
tanh2_layer = mx.symbol.Activation(data = fc2_layer, act.type = 'tanh')

tanh_list = list(tanh1_layer, tanh2_layer)

#Tip-1: The 'dim' meaning the conbining dimention, and it is in the order of 0, 1, 2,
#       ... but not 1, 2, 3, ...
#       You need to know the dimention number here is in contrast.
#       For example, if your data shape is 3-dimention, the 'dim' should be
#       in order of 2, 1, 0; if your data shape is 4-dimention, the 'dim'
#       should be in order of 3, 2, 1, 0; Now our data shape is 2-dimention and 
#       we want to conbine it by the first dimention. The 'dim' is in order of
#       1, 0 in this data, so we need to assign 1 in this example.
#Tip-2: Please use 'mx.symbol.infer.shape' function to check the output datashape.

tanh_list$dim = 1
tanh_list$num.args = 2
tanh_concat = mxnet:::mx.varg.symbol.Concat(tanh_list)

#mx.symbol.infer.shape(fc_concat, data1 = c(2, batch_size), data2 = c(2, batch_size))$out.shapes

linear_out = mx.symbol.FullyConnected(data = tanh_concat, num.hidden = 1, name = 'linear_out')

#Tip-3: Because our model is too deep that loss gradient cannot send to first 2 layers.
#       To define a small weight for first 2 layers in loss function can solve this problem.
#       The detailed example is shown in '1-5 multilayer perceptron (multi-layer)'
#       You can try 

loss_out = linear_out * 0.998 + fc1_layer * 0.001 + fc2_layer * 0.001
#loss_out = linear_out


#Try it: You can try to replace 'loss_out = linear_out' from above commond, and this trainning will fail. 

out_layer = mx.symbol.MakeLoss(mx.symbol.mean(mx.symbol.square(mx.symbol.Reshape(loss_out, shape = 0) - label)), name = 'rmse')

input_names = c('data1', 'data2', 'label')

#4. Training model

#4-1. Build an executor to train model

my_executor = mx.simple.bind(symbol = out_layer, data1 = c(2, batch_size), data2 = c(2, batch_size), ctx = mx.cpu(), grad.req = "write")

#4-2. Set the initial parameters

mx.set.seed(0)
new_arg = mxnet:::mx.model.init.params(symbol = out_layer, input.shape = list(data1 = c(2, batch_size), data2 = c(2, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
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
    cat(paste0("iteration = ", i, ": Root-Mean-Square Error (RMSE) = ", formatC(as.array(my_executor$ref.outputs$rmse_output), format = "f", 5), "\n"))
    Sys.sleep(0.1)
  }
  
}

#5. Save and load model

#5-1. Save model (Note: we need to save the 'fc_layer' but not the 'out_layer')

mx.symbol.save(linear_out, filename = "model/multi_input_regression.json")
mx.nd.save(my_executor$arg.arrays[!names(my_executor$arg.arrays)%in%input_names], filename = "model/multi_input_regression_arg_params.params")
mx.nd.save(my_executor$aux.arrays, filename = "model/multi_input_regression_aux_params.params")

#5-2. Load model

My_sym = mx.symbol.load("model/multi_input_regression.json")
My_arg_params = mx.nd.load("model/multi_input_regression_arg_params.params")
My_aux_params = mx.nd.load("model/multi_input_regression_aux_params.params")

#5-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data1 = c(2, 1000), data2 = c(2, 1000), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data1 = mx.nd.array(X1.array), data2 = mx.nd.array(X2.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

predict_Y = My_exec$ref.outputs$linear_out_output
RMSE = mean((as.array(predict_Y) - Y.array)^2)
print(RMSE)
