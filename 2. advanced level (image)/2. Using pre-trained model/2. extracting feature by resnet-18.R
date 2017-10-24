library(mxnet)
library(imager)

#1. Prepare image

#1-1. Load and display image

img <- load.image(system.file("extdata/parrots.png", package="imager"))
par(mar=rep(0,4))
plot(NA, xlim = 0:1, ylim = 0:1, xaxt = "n", yaxt = "n", bty = "n")
rasterImage(img, -0.04, -0.04, 1.04, 1.04, interpolate=FALSE)

#1-2. Resize image

preproc.image = function(im, mean.image = NULL) {
  # crop the image
  shape <- dim(im)
  if (shape[1] != shape[2]) {
    short.edge <- min(shape[1:2])
    xx <- floor((shape[1] - short.edge) / 2)
    yy <- floor((shape[2] - short.edge) / 2)
    cropped <- crop.borders(im, xx, yy)
  } else {
    cropped <- array(im, dim = c(shape, 1, 1))
    cropped <- cropped/max(cropped)
  }
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- dim(arr)[-3]
  # subtract the mean
  if (is.null(mean.image)) {mean.image = mean(arr)}
  normed <- arr - mean.image
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(dim(normed), 1)
  return(normed)
}

normed = preproc.image(img)

#2. Using resnet-18 model

#2-1. Load pre-trained model

#You can download pre-trained model from: http://data.dmlc.ml/
#Completed model includes two files: .json & .params
#This example file was download from: http://data.dmlc.ml/mxnet/models/imagenet/resnet/18-layers/

res_model = mx.model.load('resnet-18', 0)
res_sym = mx.symbol.load("model/resnet-18-symbol.json")

#2-2. Define the layer symbols

all_layers = res_sym$get.internals()

flatten0_output = which(all_layers$outputs == 'flatten0_output') %>% all_layers$get.output()
fc1_output = which(all_layers$outputs == 'fc1_output') %>% all_layers$get.output()
out = mx.symbol.Group(c(flatten0_output, fc1_output))
executor = mx.simple.bind(symbol = out, data = c(224, 224, 3, 1), ctx = mx.cpu())

#3-2. Extract features

mx.exec.update.arg.arrays(executor, res_model$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(executor, res_model$aux.params, match.name = TRUE)
mx.exec.update.arg.arrays(executor, list(data = mx.nd.array(normed)), match.name = TRUE)
mx.exec.forward(executor, is.train = FALSE)

feature = as.array(executor$ref.outputs$flatten0_output)
print(feature)
