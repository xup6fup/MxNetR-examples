library(mxnet)

#1. Generate example data (equal length data)

set.seed(0)

data(austres)

seqs = as.numeric(t(austres))
random.start = sample(1:(length(seqs)-10), 50, replace = TRUE)

seqs.array = sapply(random.start, function(x) {seqs[x+0:10]})

X.array = array(seqs.array[-11,], dim = c(10, 50))
Y.array = array(seqs.array[-1,], dim = c(10, 50))

#2. Define the model architecture

#This architecture is builded by simple recurrent unit.
#2 inputs: 1 for X; 1 for the last prediction status

batch_size = 5
seq_len = 10
num_hidden = 3

simple_recurrent_unit = function (new_data, last_status = NULL, params = NULL, first = TRUE) {
  
  if (first) {
    out = new_data
  } else {
    LIST = list(new_data, last_status)
    LIST$dim = 2
    LIST$num.args = 2
    concat_LIST = mxnet:::mx.varg.symbol.Concat(LIST)
    FC_WEIGHT = mx.symbol.FullyConnected(data = concat_LIST, num.hidden = 1,
                                         weight = params$highway_weight,
                                         bias = params$highway_bias)
    sigmoid_WEIGHT = mx.symbol.Activation(data = FC_WEIGHT, act.type = 'sigmoid')
    reshape_WEIGHT = mx.symbol.reshape(sigmoid_WEIGHT, shape = c(1, 1, 1, batch_size))
    out = mx.symbol.broadcast_mul(reshape_WEIGHT, new_data) + mx.symbol.broadcast_mul(1 - reshape_WEIGHT, last_status)
  }
  
  return(out)
}

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')

PARAMS = list(highway_weights = mx.symbol.Variable(name = 'highway_weight'),
              highway_bias = mx.symbol.Variable(name = 'highway_bias'))


#Tip-1: The squeeze_axis is in the order of 0, 1, 2, ... but not 1, 2, 3, ...
#       You need to know the dimention number here is in contrast.
#       For example, if your data shape is 3-dimention, the squeeze_axis should be
#       in order of 2, 1, 0; if your data shape is 4-dimention, the squeeze_axis
#       should be in order of 3, 2, 1, 0; Now our data shape is 2-dimention and 
#       we want to split it by the first dimention. The squeeze_axis is in order of
#       1, 0 in this data, so we need to assign 1 in this example.
#Tip-2: Please use 'mx.symbol.infer.shape' function to check the output datashape

sub_data = mx.symbol.SliceChannel(data = data, num_outputs = seq_len, squeeze_axis = 1)
#mx.symbol.infer.shape(sub_data, data = c(seq_len, batch_size))$out.shapes

output_list = list()

for (i in 1:seq_len) {
  New_data = mx.symbol.reshape(sub_data[[i]], shape = c(1, 1, 1, batch_size))
  if (i == 1) {
    output_list[[i]] = simple_recurrent_unit(new_data = New_data, params = PARAMS)
  } else {
    output_list[[i]] = simple_recurrent_unit(new_data = New_data, last_status = output_list[[i-1]], params = PARAMS, first = FALSE)
  }
  #output_list[[i]] = mx.symbol.reshape(output, shape = c(num_hidden, 1, 1, batch_size))
}

#Tip-3: The 'dim' meaning the conbining dimention, and the order here is the same as
#       above example.

output_list$dim = 2
output_list$num.args = seq_len

concat = mxnet:::mx.varg.symbol.Concat(output_list)
#mx.symbol.infer.shape(concat, data = c(seq_len, batch_size))$out.shapes

#Tip-4: Now we need to integrate 5 feature to 1 output by sequence. Convolution filter
#       can handle this situation.

fc1_seqs = mx.symbol.Convolution(data = concat, kernel = c(1, 1), stride = c(1, 1), num.filter = 1, name = 'fc1_seqs')
reshape_seqs_2 = mx.symbol.reshape(fc1_seqs, shape = c(seq_len, batch_size))
#loss_layer = mx.symbol.broadcast_minus(reshape_seqs_2, label)
out_layer = mx.symbol.LinearRegressionOutput(data = reshape_seqs_2, name = 'out_layer')

#3. Training model

mx.set.seed(0)

logger = mx.metric.logger$new()

rnn_model = mx.model.FeedForward.create(out_layer,
                                        X = X.array, y = Y.array,
                                        ctx = mx.cpu(), num.round = 20,
                                        array.batch.size = batch_size, learning.rate = 0.0000000005,
                                        momentum = 0, wd = 0, array.layout = "colmajor",
                                        eval.metric = mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(5, logger))

#4. Save and load model

#4-1. Save model

mx.model.save(rnn_model, "model/rnn_model", iteration = 0)

#4-2. Load model

My_model = mx.model.load("model/rnn_model", iteration = 0)

#4-3. Inference

predict_Y = predict(rnn_model, X.array, array.layout = "colmajor", array.batch.size = batch_size)
RMSE = mean((predict_Y - Y.array)^2)
print(RMSE)

