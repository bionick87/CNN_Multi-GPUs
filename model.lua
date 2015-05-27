-- Copyright (c) <2015>, <Nicolo' Savioli>
-- All rights reserved.

-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
-- 1. Redistributions of source code must retain the above copyright
-- notice, this list of conditions and the following disclaimer.
-- 2. Redistributions in binary form must reproduce the above copyright
-- notice, this list of conditions and the following disclaimer in the
-- documentation and/or other materials provided with the distribution.
-- 3. All advertising materials mentioning features or use of this software
-- must display the following acknowledgement:
-- This product includes software developed by the <organization>.
-- 4. Neither the name of the Nicolo'Savioli nor the
-- names of its contributors may be used to endorse or promote products
-- derived from this software without specific prior written permission.

-- THIS SOFTWARE IS PROVIDED BY NICOLO' SAVIOLI ''AS IS'' AND ANY
-- EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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


