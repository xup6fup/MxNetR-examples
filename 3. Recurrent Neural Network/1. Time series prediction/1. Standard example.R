library(mxnet)

#1. Generate example data (equal length data)

set.seed(0)

data(austres)

seqs = as.numeric(t(austres))
seqs = (seqs - mean(seqs))/sd(seqs)
random.start = sample(1:(length(seqs)-10), 50, replace = TRUE)

seqs.array = sapply(random.start, function(x) {seqs[x+0:10]})

X.array = array(seqs.array[,-11], dim = c(10, 50))
Y.array = array(seqs.array[,-1], dim = c(10, 50))

#2. Define the model architecture

#This architecture is builded by simple recurrent unit.
#2 inputs: 1 for X; 1 for the last prediction status

batch_size = 10
seq_len = 10
num_hidden = 3

simple_recurrent_unit = function (new_data, last_status = NULL, params, num_hidden, first = TRUE) {
  
  new_data2hidden = mx.symbol.FullyConnected(data = new_data, num.hidden = num_hidden,
                                             weight = params$new_data2hidden_weight,
                                             bias = params$new_data2hidden_bias)
  
  if (first) {
    sum_value = new_data2hidden
  } else {
    last_status2hidden = mx.symbol.FullyConnected(data = last_status, num.hidden = num_hidden,
                                                  weight = params$last_status2hidden_weight,
                                                  bias = params$last_status2hidden_bias)
    
    sum_value = 0.5 * new_data2hidden + 0.5 * last_status2hidden
  }
  
  out = mx.symbol.Activation(data = sum_value, act.type = 'relu')
  
  return(out)
}

data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')

PARAMS = list(new_data2hidden_weight = mx.symbol.Variable(name = 'new_data2hidden_weight'),
              new_data2hidden_bias = mx.symbol.Variable(name = 'new_data2hidden_bias'),
              last_status2hidden_weight = mx.symbol.Variable(name = 'last_status2hidden_weight'),
              last_status2hidden_bias = mx.symbol.Variable(name = 'last_status2hidden_bias'))

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
  if (i == 1) {
    output = simple_recurrent_unit(new_data = sub_data[[i]], params = PARAMS, num_hidden = num_hidden)
  } else {
    output = simple_recurrent_unit(new_data = sub_data[[i]], last_status = output_list[[i-1]],
                                   params = PARAMS, num_hidden = num_hidden, first = FALSE)
  }
  output_list[[i]] = mx.symbol.reshape(output, shape = c(num_hidden, 1, 1, batch_size))
}

#Tip-3: The 'dim' meaning the conbining dimention, and the order here is the same as
#       above example.

output_list$dim = 2
output_list$num.args = seq_len

concat = mxnet:::mx.varg.symbol.Concat(output_list)
#mx.symbol.infer.shape(concat, data = c(seq_len, batch_size))$out.shapes

#Tip-4: Now we need to integrate 5 feature to 1 output by sequence. Convolution filter
#       can handle this situation.

fc1_seqs = mx.symbol.Convolution(data = concat, kernel = c(num_hidden, 1), stride = c(1, 1), num.filter = 5, name = 'fc1_seqs')
relu1_seqs = mx.symbol.Activation(data = fc1_seqs, act.type = 'relu', name = 'relu1_seqs')
fc2_seqs = mx.symbol.Convolution(data = relu1_seqs, kernel = c(1, 1), stride = c(1, 1), num.filter = 1, name = 'fc2_seqs')

reshape_seqs = mx.symbol.reshape(fc2_seqs, shape = c(seq_len, batch_size))
out_layer = mx.symbol.LinearRegressionOutput(data = reshape_seqs, label = label, name = 'out_layer')

#3. Training model

mx.set.seed(0)

logger = mx.metric.logger$new()

rnn_model = mx.model.FeedForward.create(out_layer,
                                        X = X.array, y = Y.array,
                                        ctx = mx.cpu(), num.round = 200,
                                        array.batch.size = batch_size, learning.rate = 0.01,
                                        momentum = 0.9, wd = 0.001, array.layout = "colmajor",
                                        eval.metric = mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(5, logger))

#4. Save and load model

#4-1. Save model

#mx.model.save(lr_model, "model/linear_regression", iteration = 0)

#4-2. Load model

#My_model = mx.model.load("model/linear_regression", iteration = 0)

#4-3. Inference

#predict_Y = predict(rnn_model, X.array, array.layout = "colmajor", array.batch.size = batch_size)
#RMSE = mean((as.array(predict_Y) - Y)^2)
#print(RMSE)

