-- <Nicolo Savioli> --

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
require 'fbcunn'
require 'fbnn'
require 'xlua'

-----------------------------------------------------------
-- a typical modern convolution network (conv+relu+pool) --
-----------------------------------------------------------

--------------------
-- CNN input size --
--------------------

sample_size  = {3, 32, 32}

-----------------------
-- feature maps size --
-----------------------

feature_size = {3, 32, 64}
filter_size  = {5, 5}
pool_size    = {2, 2}
pool_step    = {2, 2}
classifer_hidden_units = {512}

----------------------------------------------------------------------
-- WARNING: change this if you change feature/filter/pool size/step --
----------------------------------------------------------------------

features_out = feature_size[3] * 5 * 5

-------------
-- Dropout --
-------------

dropout_p = 0.5

if opt.dropout then
  print("Using dropout ...")
end

---------------------------
-- Configuring optimizer --
---------------------------

optim_state = {
  learningRate = opt.learning_rate,
  learningRateDecay = 0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weight_decay,
}


print 'Defining modern convolution network (conv+relu+pool) [2 Stages]'

model = nn.Sequential()

-------------
-- Stage 1 --
-------------

model:add(cudnn.SpatialConvolution(feature_size[1], feature_size[2], filter_size[1], filter_size[2]), 1, 1) 
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) 

-------------
-- Stage 2 --
-------------

model:add(cudnn.SpatialConvolution(feature_size[2], feature_size[3], filter_size[1], filter_size[2]), 1, 1) 
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) 

------------------------------------------------
-- Get feature vectors i.e. flat feature maps --
------------------------------------------------

model:add(nn.Reshape(features_out, true)) 
if opt.dropout then
  model:add(nn.Dropout(dropout_p))
end

----------------------------
-- Fully connected layers --
----------------------------

model:add(nn.Linear(features_out, classifer_hidden_units[1]))
model:add(nn.ReLU())
if opt.dropout then
  model:add(nn.Dropout(dropout_p))
end

model:add(nn.Linear(classifer_hidden_units[1], 10))


