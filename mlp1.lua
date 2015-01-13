require 'torch'
require 'nn'
require './lib/util'
require './lib/train'

local function create_model()
  torch.manualSeed(config.model_seed)
  local model = nn.Sequential()
  model:add(nn.Reshape(INPUT_SZ[2] * INPUT_SZ[3]))
  model:add(nn.Linear(INPUT_SZ[2] * INPUT_SZ[3], 2048))
  model:add(nn.Tanh())
  model:add(nn.Linear(2048, #CLASSES))
  model:add(nn.LogSoftMax())
  local criterion = nn.DistKLDivCriterion()
  
  model:cuda()
  criterion:cuda()
  return model, criterion
end

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

config = {
  id = 'mlp1_val', 
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
local model, criterion = create_model(config.model_seed)
-- epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.id = string.gsub(config.id, '_val$', '')
model, criterion = create_model(config.model_seed)
-- model = train(model, criterion, learning_rates, seeds, epochs)

config.id = string.gsub(config.id, '_val$', '')
local model = torch.load(string.format('model/%s.model', config.id))
gen_predictions(model)