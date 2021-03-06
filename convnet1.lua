require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
require 'fbcunn'
FB = true

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

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

config = {
  id = 'convnet1_val', 
  learningRateDecay = 1.0e-6, 
  momentum = 0.9, 
  batch_size = 128, 
  model_seed = 1, 
  train_jitter = false, 
  test_jitter = false, 
  early_stop = 6
}

local learning_rates = {1.0, 0.1}
local seeds = {2, 3}
local epochs = {50, 50}
local val_sz = 6144
local model, criterion = create_model()
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.id = string.gsub(config.id, '_val$', '')
model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

-- config.id = string.gsub(config.id, '_val$', '')
-- local model = torch.load(string.format('model/%s.model', config.id))
gen_predictions(model)