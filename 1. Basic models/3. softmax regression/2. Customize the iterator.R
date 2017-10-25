library(mxnet)

#1. Call example data (iris)

data(iris)

X.array = array(as.matrix(iris[,-5]), dim = c(4, 150))
Y.array = array(t(model.matrix(~ -1 + iris[,5])), dim = c(3, 150))

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
    idx = sample(dim(Y.array)[2], size = batch_size, replace = TRUE)
    data = mx.nd.array(X.array[,idx])
    label = mx.nd.array(Y.array[,idx])
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iter = my_iterator(batch_size)

#3. Define the model architecture (m-log loss function)

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer = mx.symbol.FullyConnected(data = data, num.hidden = 3, name = 'fc_layer')
softmax_layer = mx.symbol.SoftmaxOutput(data = fc_layer, label = label, name = 'sofmax_layer')
m_logloss = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(softmax_layer), label))
out_layer = mx.symbol.MakeLoss(m_logloss, name = 'm_logloss')

input_names = mxnet:::mx.model.check.arguments(out_layer)

#4. Training model

#4-1. Build an executor to train model

my_executor = mx.simple.bind(symbol = out_layer, data = c(4, batch_size), label = c(3, batch_size), ctx = mx.cpu(), grad.req = "write")

#4-2. Set the initial parameters

new_arg = mxnet:::mx.model.init.params(symbol = out_layer, input.shape = list(data = c(4, batch_size), label = c(3, batch_size)), output.shape = NULL, initializer = mxnet:::mx.init.uniform(0.01), ctx = mx.cpu())
mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

#4-3. Define the optimizer

my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.1, momentum = 0, wd = 0, rescale.grad = 1, clip_gradient = 1)
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
    cat(paste0("iteration = ", i, ": m-logloss = ", formatC(as.array(my_executor$ref.outputs$m_logloss_output), format = "f", 5), "\n"))
    Sys.sleep(0.1)
  }
  
}

#5. Save and load model

#5-1. Save model (Note: we need to save the 'logistic_layer' but not the 'out_layer')

mx.symbol.save(softmax_layer, filename = "model/softmax_regression.json") 
mx.nd.save(my_executor$arg.arrays[!names(my_executor$arg.arrays)%in%input_names], filename = "model/softmax_regression_arg_params.params")
mx.nd.save(my_executor$aux.arrays, filename = "model/softmax_regression_aux_params.params")

#5-2. Load model

My_sym = mx.symbol.load("model/softmax_regression.json")
My_arg_params = mx.nd.load("model/softmax_regression_arg_params.params")
My_aux_params = mx.nd.load("model/softmax_regression_aux_params.params")

#5-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data = c(4, 150), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data = mx.nd.array(X.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

predict_Y = as.array(My_exec$ref.outputs$sofmax_layer_output)
predict_label = max.col(t(predict_Y))
confusion_table = table(predict_label, iris[,5])
print(confusion_table)