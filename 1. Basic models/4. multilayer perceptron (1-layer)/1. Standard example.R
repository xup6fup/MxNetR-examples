library(mxnet)

#1. Call example data (iris)

data(iris)

X.array = array(as.matrix(iris[,-5]), dim = c(4, 150))
Y.array = array(t(model.matrix(~ -1 + iris[,5])), dim = c(3, 150))

#2. Define the model architecture

#Tip: for the deeper and complex model, you can use 'mx.symbol.infer.shape' function to check the output datashape

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 12, name = 'fc_layer_1')
sigmoid_layer_1 = mx.symbol.Activation(data = fc_layer_1, act.type = 'sigmoid', name = 'sigmoid_layer_1')
#mx.symbol.infer.shape(fc_layer_1, data = c(4, 150))$out.shapes
fc_layer_2 = mx.symbol.FullyConnected(data = sigmoid_layer_1, num.hidden = 3, name = 'fc_layer_2')
#mx.symbol.infer.shape(fc_layer_2, data = c(4, 150))$out.shapes
out_layer = mx.symbol.SoftmaxOutput(data = fc_layer_2, label = label, name = 'out_layer')
#mx.symbol.infer.shape(out_layer, data = c(4, 150), label = c(3, 150))$out.shapes

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

#3-2. Training

mx.set.seed(0)

logger = mx.metric.logger$new()

mlp_model = mx.model.FeedForward.create(out_layer,
                                             X = X.array, y = Y.array,
                                             ctx = mx.cpu(), num.round = 100,
                                             array.batch.size = 50, learning.rate = 0.1,
                                             momentum = 0, wd = 0, array.layout = "colmajor",
                                             eval.metric = my.eval.metric.mlogloss,
                                             epoch.end.callback = mx.callback.log.train.metric(5, logger))

#4. Save and load model

#4-1. Save model

mx.model.save(mlp_model, "model/multilayer_perceptron", iteration = 0)

#4-2. Load model

My_model = mx.model.load("model/multilayer_perceptron", iteration = 0)

#4-3. Inference

predict_Y = predict(My_model, X.array, array.layout = "colmajor")
predict_label = max.col(t(predict_Y))
confusion_table = table(predict_label, iris[,5])
print(confusion_table)

