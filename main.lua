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

----------------------------
-- Command line arguments --
----------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Multi CNN-GPUs:')
cmd:text()
cmd:text  ('Options:')
cmd:option('-seed',            1,           'fixed input seed for repeatable experiments')
cmd:option('-threads',         20,          'number of threads')
cmd:option('-gpuid',           1,           'gpu id')
cmd:option('-train_criterion', 'NLL',       'train_criterion: MSE | NLL')
cmd:option('-learning_rate',   5e-2,        'learning rate at t=0')
cmd:option('-momentum',        0.6,         'momentum')
cmd:option('-weight_decay',    1e-5,        'weight decay')
cmd:option('-batch_size',      500,         'mini-batch size (1 = pure stochastic)')
cmd:option('-epoch_size',      50,          'number of batches per epoch')
cmd:option('-dropout',         false,       'do dropout with 0.5 probability')
cmd:option('-print_layers_op', false,       'Output the values from each layers')
cmd:option('-trsize'         , 10,          'Dataset size')
cmd:option('-type',            'cuda',      'type: double | float | cuda')
cmd:option('-size',            'full',      'how many samples do we load: small | full')
cmd:option('-statinterval',    50,          'How many epoch iteration you want save matrix')
cmd:text()
opt = cmd:parse(arg)

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'


dofile 'config.lua'
dofile 'data_cifar10.lua' 
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'
----------------------------------------------------------------------
