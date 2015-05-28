-- <Nicolo' Savioli> --

require 'torch'
require 'nn'
require 'optim'
require 'xlua'


------------------
-- Output model --
------------------

if opt.train_criterion == 'MSE' then
  model:add(nn.SoftMax())
  criterion = nn.MSECriterion()
elseif opt.train_criterion == 'NLL' then
  model:add(nn.LogSoftMax())
  criterion = nn.ClassNLLCriterion() 
end

-----------------------------
-- Set GPUs system options --
-----------------------------

torch.setdefaulttensortype('torch.FloatTensor')

if opt.type == 'cuda' then 
   require 'cutorch' 
   require 'cunn'
   require 'cudnn'
   require 'fbcunn'
   require 'fbnn'
   cutorch.setDevice(opt.gpuid)
   nGPUs = cutorch.getDeviceCount()
end
----------------------
-- ConfusionMatrtix --
----------------------

confusion = optim.ConfusionMatrix(classes)

----------------
-- Multi-GPUs --
----------------

if opt.type == 'cuda' then
    if nGPUs > 1 then
       print('Using data parallel')
       local gpu_net = nn.DataParallel(1):cuda()
       for i = 1,nGPUs do
           print ('==> Multi-GPUs: setting Model in ' .. cutorch.getDeviceProperties(i).name)
           local cur_gpu = math.fmod(opt.gpuid + (i-1)-1, cutorch.getDeviceCount())+1
           cutorch.setDevice(cur_gpu)
           gpu_net:add(model:clone():cuda(), cur_gpu)
        end
        cutorch.setDevice(opt.gpuid)
        model = gpu_net
    end
end
------------------------
--   model/criterion  --
------------------------

if opt.type     == 'float' then model = model:float() criterion = criterion:float()  
elseif opt.type == 'cuda' then model = model:cuda() criterion = criterion:cuda() end

---------------
-- Optimizer --
---------------

optimator = nn.Optim(model, optim_state)

-------------------------------------------------------------------
-- The the tensor variables for model params and gradient params --
------------------------------------------------------------------- 

 if opt.type == 'cuda' then
   if nGPUs > 1 then
         params, grad_params = model:get(1):getParameters()
         optimator:setParameters(optim_state)
         cutorch.synchronize()
         -- set the dropouts to training mode
         model:training()
         model:cuda()  -- get it back on the right GPUs
      else
      params, grad_params = model:getParameters()
      -- set the dropouts to training mode
      model:training()
   end

 else
     params, grad_params = model:getParameters()
     -- set the dropouts to training mode
     model:training()
 end

----------------------------------------------------------------
-- Define the function for gradient optimization i.e. for SGD --
----------------------------------------------------------------

local function train()

  -- shuffle at each epoch --
  shuffle = torch.randperm(trsize)
  for t = 1,trainData:size(), opt.batch_size do
      if (t+ opt.batch_size-1) <= trainData:size() then
         local k=1
 
         -- disp progress --
         xlua.progress(t, trainData:size())

         -- create mini batch --
         input  = torch.Tensor(opt.batch_size,sample_size[1],sample_size[2],sample_size[3])
         target = torch.Tensor(opt.batch_size)

         for i = t,math.min(t+ opt.batch_size-1,trainData:size()) do
              -- load new sample --
              input [k]    = trainData.data[shuffle[i]]
              target[k]    = trainData.labels[shuffle[i]]
              if opt.type == 'float' then input  = input:float() target = target:float()
              elseif opt.type == 'cuda' then input = input:cuda() target = target:cuda() end
            k=k+1
         end

         f, output = optimator:optimize(optim.sgd,input,target,criterion)
         if nGPUs > 1 then cutorch.synchronize() end
         for i=1,opt.batch_size do
             confusion:add(output[i],target[i])
         end
         -- print confusion matrix --
         print(confusion)
         -- next epoch --
         confusion:zero()
      end
   end
end -- END: local function train()


-- Train- epoch -- 
print '==> defining training procedure'
local tm = torch.Timer()
for i=1,opt.epoch_size do
     train()
     if math.fmod(i,opt.statinterval) == 0 then
        torch.save(paths.concat(path_save, 'model' .. '.t7'), model)
     end
end
print '==>  Saving model...'
torch.save(paths.concat(path_save, 'model' .. '.t7'), model)
print('Time took: ' .. tm:time().real)
print('Training finish.')

-- EOF -- 
