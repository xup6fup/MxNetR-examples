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

###Now we need to solve the vanishing gradient problem by a special network architecture.
###The main idea is to build a direct connection between each layer and output.
###It can help us to transfer classification error to first two layers.

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')

fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 6, name = 'fc_layer_1')
sigmoid_layer_1 = mx.symbol.Activation(data = fc_layer_1, act.type = 'sigmoid', name = 'sigmoid_layer_1')
out_layer_1 = mx.symbol.FullyConnected(data = sigmoid_layer_1, num.hidden = 3, name = 'out_layer_1')
softmax_layer_1 = mx.symbol.SoftmaxOutput(data = out_layer_1, label = label, name = 'softmax_layer_1')

fc_layer_2 = mx.symbol.FullyConnected(data = sigmoid_layer_1, num.hidden = 8, name = 'fc_layer_2')
sigmoid_layer_2 = mx.symbol.Activation(data = fc_layer_2, act.type = 'sigmoid', name = 'sigmoid_layer_2')
out_layer_2 = mx.symbol.FullyConnected(data = sigmoid_layer_2, num.hidden = 3, name = 'out_layer_2')
softmax_layer_2 = mx.symbol.SoftmaxOutput(data = out_layer_2, label = label, name = 'softmax_layer_2')

fc_layer_3 = mx.symbol.FullyConnected(data = sigmoid_layer_2, num.hidden = 10, name = 'fc_layer_3')
sigmoid_layer_3 = mx.symbol.Activation(data = fc_layer_3, act.type = 'sigmoid', name = 'sigmoid_layer_3')
out_layer_3 = mx.symbol.FullyConnected(data = sigmoid_layer_3, num.hidden = 3, name = 'out_layer_3')
softmax_layer_3 = mx.symbol.SoftmaxOutput(data = out_layer_3, label = label, name = 'softmax_layer_3')

conbine_layer = 0.001 * softmax_layer_1 + 0.049 * softmax_layer_2 + 0.950 * softmax_layer_3

m_logloss = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(conbine_layer), label))
loss_layer = mx.symbol.MakeLoss(m_logloss, name = 'm_logloss')

input_names = mxnet:::mx.model.check.arguments(loss_layer)

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

#5. Save and load model

#5-1. Save model (we only want to save the result only calculating by softmax_layer_standard)

mx.symbol.save(softmax_layer_3, filename = "model/multilayer_perceptron.json") 
mx.nd.save(my_executor$arg.arrays[!names(my_executor$arg.arrays)%in%c(input_names, 'out_layer_1_weight', 'out_layer_1_bias', 'out_layer_2_weight', 'out_layer_2_bias')], filename = "model/multilayer_perceptron_arg_params.params")
mx.nd.save(my_executor$aux.arrays, filename = "model/multilayer_perceptron_aux_params.params")

#5-2. Load model

My_sym = mx.symbol.load("model/multilayer_perceptron.json")
My_arg_params = mx.nd.load("model/multilayer_perceptron_arg_params.params")
My_aux_params = mx.nd.load("model/multilayer_perceptron_aux_params.params")

#5-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data = c(4, 150), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data = mx.nd.array(X.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

predict_Y = as.array(My_exec$ref.outputs$softmax_layer_3_output)
predict_label = max.col(t(predict_Y))
confusion_table = table(predict_label, iris[,5])
print(confusion_table)

#6. Visualize vanishing gradient problem

###After model saving and loading, let us re-train this model
####to see the mean gradient in each iteration agian.

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
        plot(1:i, mean_grad_list[,j], type = 'l', xlim = c(1, 3000), ylim = c(0, max(mean_grad_list)),
             xlab = 'iteration', ylab = 'mean(abs(gradients))', col = col_list[j])
      } else {
        lines(1:i, mean_grad_list[,j], col = col_list[j])
      }
    }
    legend('topright', colnames(mean_grad_list), col = col_list, lwd = 1, cex = 0.5)
    
    Sys.sleep(1)
  }
  
}

###Is it really cool?
