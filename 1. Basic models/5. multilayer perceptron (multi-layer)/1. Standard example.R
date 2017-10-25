library(mxnet)

#1. Call example data (iris)

data(iris)

X.array = array(as.matrix(iris[,-5]), dim = c(4, 150))
Y.array = array(t(model.matrix(~ -1 + iris[,5])), dim = c(3, 150))

#2. Define the model architecture

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

#3. Training model

#3-1. Define loss function: m-log loss

my.eval.metric.mlogloss <- mx.metric.custom(
  name = "m-logloss", 
  function(real, pred) {
    real1 = as.numeric(real)
    pred1 = as.numeric(pred)
    pred1[pred1 <= 1e-6] = 1e-6
    pred1[pred1 >= 1 - 1e-6] = 1 - 1e-6
    return(-mean(real1 * log(pred1), na.rm = TRUE))
  }
)

#3-2. Training (for a deeper model, we often need more step for training it.)

mx.set.seed(0)

logger = mx.metric.logger$new()

mlp_model = mx.model.FeedForward.create(conbine_layer,
                                        X = X.array, y = Y.array,
                                        ctx = mx.cpu(), num.round = 1000,
                                        array.batch.size = 50, learning.rate = 0.3,
                                        momentum = 0, wd = 0, array.layout = "colmajor",
                                        eval.metric = my.eval.metric.mlogloss,
                                        epoch.end.callback = mx.callback.log.train.metric(5, logger))

#4. Save and load model

#4-1. Save model (we only want to save the result only calculating by out_layer_3)

mx.symbol.save(softmax_layer_3, filename = "model/multilayer_perceptron.json") 
mx.nd.save(mlp_model$arg.params[!names(mlp_model$arg.params)%in%c('out_layer_1_weight', 'out_layer_1_bias', 'out_layer_2_weight', 'out_layer_2_bias')], filename = "model/multilayer_perceptron_arg_params.params")
mx.nd.save(mlp_model$aux.params, filename = "model/multilayer_perceptron_aux_params.params")

#4-2. Load model

My_sym = mx.symbol.load("model/multilayer_perceptron.json")
My_arg_params = mx.nd.load("model/multilayer_perceptron_arg_params.params")
My_aux_params = mx.nd.load("model/multilayer_perceptron_aux_params.params")

#4-3. Inference

My_exec = mx.simple.bind(symbol = My_sym, data = c(4, 150), ctx = mx.cpu(), grad.req = "null")
mx.exec.update.arg.arrays(My_exec, My_arg_params, match.name = TRUE)
mx.exec.update.aux.arrays(My_exec, My_aux_params, match.name = TRUE)
mx.exec.update.arg.arrays(My_exec, list(data = mx.nd.array(X.array)), match.name = TRUE)
mx.exec.forward(My_exec, is.train = TRUE)

predict_Y = as.array(My_exec$ref.outputs$softmax_layer_3_output)
predict_label = max.col(t(predict_Y))
confusion_table = table(predict_label, iris[,5])
print(confusion_table)
