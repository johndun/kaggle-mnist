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
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 1, 64, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 64, 64, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))

  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 64, 128, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 128, 128, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))

  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 128, 256, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 256, 256, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))

  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 256, 512, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  add_conv_layer(model, 512, 512, 3, 3, 1, 1)
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.25))

  add_conv_layer(model, 512, 1024, 3, 3, 1, 1)
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
  id = 'convnet3_val', 
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
local val_sz = 6144
local model, criterion = create_model(config.model_seed)
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.id = string.gsub(config.id, '_val$', '')
model, criterion = create_model(config.model_seed)
model = train(model, criterion, learning_rates, seeds, epochs)

-- config.id = string.gsub(config.id, '_val$', '')
-- local model = torch.load(string.format('model/%s.model', config.id))
gen_predictions(model)