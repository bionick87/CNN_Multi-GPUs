-- <Nicolo Savioli> --


require 'torch'
require 'nn'
require 'optim'
require 'xlua' 


 if opt.type == 'cuda' then
    require 'cutorch' 
    require 'cunn'
    require 'cudnn'
    require 'fbcunn'
    require 'fbnn'
 end
 


local function validation()
  
     print("---------- 	Testing:    -----------")
     result = optim.ConfusionMatrix(classes)   
     input  = testData.data
     input  = input:cuda()
     target = testData.labels
     pred   = model:forward(input)
     if nGPUs > 1 then cutorch.synchronize() end
     for i=1,testData:size() do
         result:add(pred[i],target[i])
     end 
     testAccuracy = result.totalValid * 100
     print(result)
     
end -- END: local function validation()


local tm = torch.Timer()
validation()
print('Time took: ' .. tm:time().real)
print('Testing finish.')
