library(mxnet)

#1. Generate example data

set.seed(0)

X1 = rnorm(1000) 
X2 = rnorm(1000)
LR = X1 * 2 + X2 * 3 + rnorm(1000)
PROP = 1/(1+exp(-LR))
Y = as.integer(PROP > runif(1000))

X.array = array(rbind(X1, X2), dim = c(2, 1000))

#2. Define the model architecture

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer = mx.symbol.FullyConnected(data = data, num.hidden = 1, name = 'fc_layer')
out_layer = mx.symbol.LogisticRegressionOutput(data = fc_layer, label = label, name = 'out_layer')

#3. Training model

#3.1. Define loss function: Cross-Entropy

my.eval.metric.CE <- mx.metric.custom(
  name = "Cross-Entropy (CE)", 
  function(real, pred) {
    real1 = as.numeric(real)
    pred1 = as.numeric(pred)
    pred1[pred1 <= 1e-6] = 1e-6
    pred1[pred1 >= 1 - 1e-6] = 1 - 1e-6
    return(-mean(real1 * log(pred1) + (1 - real1) * log(1 - pred1), na.rm = TRUE))
  }
)

#3.2. Training

logger = mx.metric.logger$new()

logistic_model = mx.model.FeedForward.create(out_layer,
                                             X = X.array, y = Y,
                                             ctx = mx.cpu(), num.round = 50,
                                             array.batch.size = 100, learning.rate = 0.1,
                                             momentum = 0, wd = 0, array.layout = "colmajor",
                                             eval.metric = my.eval.metric.CE,
                                             epoch.end.callback = mx.callback.log.train.metric(5, logger))


#4. Save and load model

#4-1. Save model

mx.model.save(logistic_model, "model/logistic_regression", iteration = 0)

#4-2. Load model

My_model = mx.model.load("model/logistic_regression", iteration = 0)

#5-3. Inference

predict_Y = predict(My_model, X.array, array.layout = "colmajor")
confusion_table = table(predict_Y > 0.5, Y)
print(confusion_table)

