require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
require 'cutorch'
cutorch.setDevice(2)
-- require 'fbcunn'
-- FB = true
-- require 'cuda-convnet2'
-- CCN2 = true
TRAIN_FNAME = 'data/train_yl.t7'
TEST_FNAME = 'data/test_yl.t7'

local function create_model()
  torch.manualSeed(config.model_seed)
  local model = nn.Sequential() 
  add_conv_layer(model, 1, 128, 5, 5, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  add_conv_layer(model, 128, 256, 5, 5, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 256, 512, 4, 4, 1, 1)
  model:add(nn.ReLU())
  add_conv_layer(model, 512, 1024, 2, 2, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  add_conv_layer(model, 1024, #CLASSES, 1, 1, 1, 1)
  model:add(nn.Reshape(10))
  model:add(nn.LogSoftMax())
  local criterion = nn.DistKLDivCriterion()
  return model:cuda(), criterion:cuda()
end

-- local function create_model()
  -- torch.manualSeed(config.model_seed)
  -- local model = nn.Sequential()
  -- model:add(nn.Reshape(INPUT_SZ[2] * INPUT_SZ[3]))
  -- model:add(nn.Linear(INPUT_SZ[2] * INPUT_SZ[3], 2048))
  -- model:add(nn.Tanh())
  -- model:add(nn.Linear(2048, #CLASSES))
  -- model:add(nn.LogSoftMax())
  -- local criterion = nn.DistKLDivCriterion()
  
  -- model:cuda()
  -- criterion:cuda()
  -- return model, criterion
-- end

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

config = {
  id = 'convnet3_yl_val', 
  learningRateDecay = 1.0e-6, 
  momentum = 0.9, 
  batch_size = 128, 
  model_seed = 1, 
  train_jitter = true, 
  test_jitter = true, 
  early_stop = 12, 
  evaluate_every = 2
}

local learning_rates = {1.0, 0.1}
local seeds = {2, 3}
local epochs = {100, 100}
local val_sz = 10000
local model, criterion = create_model()
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.early_stop = false
config.id = string.gsub(config.id, '_val$', '')
model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

-- config.id = string.gsub(config.id, '_val$', '')
-- local model = torch.load(string.format('model/%s.model', config.id))
gen_predictions(model)