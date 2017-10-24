library(mxnet)

#1. Generate example data

set.seed(0)

X1 = rnorm(1000) 
X2 = rnorm(1000) 
Y = X1 * 0.7 + X2 * 1.3 - 3.1 + rnorm(1000)

X.array = array(rbind(X1, X2), dim = c(2, 1000))

#2. Define the model architecture

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer = mx.symbol.FullyConnected(data = data, num.hidden = 1, name = 'fc_layer')
out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer, label = label, name = 'out_layer')

#3. Training model

logger = mx.metric.logger$new()

lr_model = mx.model.FeedForward.create(out_layer,
                                       X = X.array, y = Y,
                                       ctx = mx.cpu(), num.round = 100,
                                       array.batch.size = 100, learning.rate = 0.005,
                                       momentum = 0, wd = 0, array.layout = "colmajor",
                                       eval.metric = mx.metric.rmse,
                                       epoch.end.callback = mx.callback.log.train.metric(5, logger))


#4. Save and load model

#4-1. Save model

mx.model.save(lr_model, "model/linear_regression", iteration = 0)

#4-2. Load model

My_model = mx.model.load("model/linear_regression", iteration = 0)

#5-3. Inference

predict_Y = predict(My_model, X.array, array.layout = "colmajor")
RMSE = mean((as.array(predict_Y) - Y)^2)
print(RMSE)

