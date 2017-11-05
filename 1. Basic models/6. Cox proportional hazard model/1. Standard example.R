library(mxnet)
library(survival)

#1. Generate example data

data(ovarian)
base_model = survfit(Surv(futime, fustat) ~ 0, data = ovarian)
important_pos = unique(c(which(base_model$n.event!=0), length(base_model$n.event)))
base_surv = (1 - base_model$n.event/base_model$n.risk)[important_pos]
base_time = base_model$time[important_pos]

X.array = array(t(as.matrix(ovarian[,c('age', 'rx')])), dim = c(2, nrow(ovarian)))

Y.array = array(1, dim = c(length(base_time), nrow(ovarian)))
rownames(Y.array) = base_time

for (i in 1:nrow(ovarian)) {
  pos = which((base_time - ovarian[i,'futime'])>=0)[1]
  if (ovarian[i,'fustat']==1) {
    Y.array[pos:length(base_time),i] = 0
  } else {
    Y.array[pos:length(base_time),i] = cumprod(base_surv[pos:length(base_time)])
  }
}

#2. Define the model architecture

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
lp_layer = mx.symbol.FullyConnected(data = data, num.hidden = 1, no.bias = TRUE, name = 'lp_layer')
hr_layer = mx.symbol.exp(data = lp_layer, name = 'hr_layer')
haz_layer = mx.symbol.FullyConnected(data = hr_layer, num.hidden = length(base_time), no.bias = TRUE, name = 'haz_layer')
relu_layer = mx.symbol.Activation(data = haz_layer, act.type = 'relu')
out_layer = mx.symbol.LinearRegressionOutput(data = relu_layer, label = label, name = 'out_layer')
#mx.symbol.infer.shape(out_layer, data = c(1, 10))$out.shapes

#3. Training model

#3-1. Define loss function: Cross-Entropy

my.eval.metric.CE <- mx.metric.custom(
  name = "Cross-Entropy (CE)", 
  function(real, pred) {
    real.array = as.array(real)
    pred.array = as.array(pred)
    cum_pred.array = apply(pred.array, 2, cumsum)
    cum_pred.array[cum_pred.array == 0] = 1e-9
    surv_pred.array = 1 - exp(-cum_pred.array)
    CE = log(surv_pred.array) * real.array + log(1 - surv_pred.array) * (1 - real.array)
    return(-mean(CE, na.rm = TRUE))
  }
)

#3-2. Training

mx.set.seed(0)

logger = mx.metric.logger$new()

cox_model = mx.model.FeedForward.create(out_layer,
                                        X = X.array, y = Y.array,
                                        ctx = mx.cpu(), num.round = 1000,
                                        array.batch.size = 10, learning.rate = 0.00001,
                                        momentum = 0, wd = 0, array.layout = "colmajor",
                                        eval.metric = my.eval.metric.CE,
                                        arg.params = new_arg$arg.params,
                                        epoch.end.callback = mx.callback.log.train.metric(5, logger))

#new_arg$arg.params$lp_layer_weight = cox_model$arg.params$lp_layer_weight

#4. Save and load model

#4-1. Save model

#mx.model.save(logistic_model, "model/logistic_regression", iteration = 0)

#4-2. Load model

#My_model = mx.model.load("model/logistic_regression", iteration = 0)

#4-3. Inference

predict_Y = predict(cox_model, X.array, array.layout = "colmajor")
survival_table = exp(-apply(predict_Y, 2, cumsum))