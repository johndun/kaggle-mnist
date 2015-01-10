require 'torch'
require 'nn'
require '../util'
require '../train'
require '../sgd'

local config = {
  learningRate = 1.0, 
  learningRateDecay = 1.0e-6, 
  momentum = 0.9, 
  batch_size = 128, 
  epochs = 20, 
  model_seed = 1, 
  train_seed = 2, 
  eval = true, 
  jitter = false
}

local function create_model()
  torch.manualSeed(config.model_seed)
  local model = nn.Sequential()
  model:add(nn.Reshape(576))
  model:add(nn.Linear(576, 2048))
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

local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
local model, criterion = create_model(config.model_seed)

-- Crop the images
train_x = batch_sample{src       = train_x, 
                       crp_off_x = 5, 
                       crp_off_y = 5, 
                       crp_sz_x  = 24,
                       crp_sz_y  = 24}
-- Split into training/validation
local test_x, test_y
train_x, train_y, test_x, test_y = validation_split(train_x, train_y, 6144)

-- Global contrast normalization
local preprocess_params = preprocess(train_x)
preprocess(test_x, preprocess_params)

torch.manualSeed(config.train_seed)
train(model, criterion, config, train_x, train_y, test_x, test_y)