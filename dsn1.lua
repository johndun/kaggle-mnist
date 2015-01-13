require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
require 'fbcunn'
FB = true

function create_model()
  local model = nn.Sequential()

  layer1 = nn.Sequential()
  layer1:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer1:add(nn.SpatialConvolutionCuFFT(1, 64, 3, 3, 1, 1))  
  layer1:add(nn.ReLU())
  layer1:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer1:add(nn.SpatialConvolutionCuFFT(64, 64, 3, 3, 1, 1))
  layer1:add(nn.ReLU())
  layer1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  layer1:add(nn.Dropout(0.5))

  model:add(layer1)


  local layer2 = nn.Sequential()
  layer2:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer2:add(nn.SpatialConvolutionCuFFT(64, 128, 3, 3, 1, 1))  
  layer2:add(nn.ReLU())
  layer2:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer2:add(nn.SpatialConvolutionCuFFT(128, 128, 3, 3, 1, 1))
  layer2:add(nn.ReLU())
  layer2:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  layer2:add(nn.Dropout(0.5))

  local layer2_ = nn.Sequential()
  layer2_:add(nn.SpatialConvolutionCuFFT(64, 10, 12, 12, 1, 1))
  layer2_:add(nn.Reshape(10))

  local split2 = nn.ConcatTable()
  split2:add(layer2)
  split2:add(layer2_)

  model:add(split2)


  local layer3 = nn.Sequential()
  layer3:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer3:add(nn.SpatialConvolutionCuFFT(128, 256, 3, 3, 1, 1))
  layer3:add(nn.ReLU())
  layer3:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer3:add(nn.SpatialConvolutionCuFFT(256, 256, 3, 3, 1, 1))
  layer3:add(nn.ReLU())
  layer3:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  layer3:add(nn.Dropout(0.5))

  local layer3_ = nn.Sequential()
  layer3_:add(nn.SpatialConvolutionCuFFT(128, 10, 6, 6, 1, 1))
  layer3_:add(nn.Reshape(10))

  local split3 = nn.ConcatTable()
  split3:add(layer3)
  split3:add(layer3_)

  local pass3 = nn.ParallelTable()
  pass3:add(split3)
  pass3:add(nn.Identity())

  model:add(pass3)
  model:add(nn.FlattenTable())


  local layer4 = nn.Sequential()
  layer4:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer4:add(nn.SpatialConvolutionCuFFT(256, 512, 3, 3, 1, 1))
  layer4:add(nn.ReLU())
  layer4:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  layer4:add(nn.SpatialConvolutionCuFFT(512, 512, 3, 3, 1, 1))
  layer4:add(nn.ReLU())
  layer4:add(nn.Dropout(0.5))

  local layer4_ = nn.Sequential()
  layer4_:add(nn.SpatialConvolutionCuFFT(256, 10, 3, 3, 1, 1))
  layer4_:add(nn.Reshape(10))

  local split4 = nn.ConcatTable()
  split4:add(layer4)
  split4:add(layer4_)

  local pass4 = nn.ParallelTable()
  pass4:add(split4)
  pass4:add(nn.Identity())
  pass4:add(nn.Identity())

  model:add(pass4)
  model:add(nn.FlattenTable())


  local layer5 = nn.Sequential()
  layer5:add(nn.SpatialConvolutionCuFFT(512, 1024, 3, 3, 1, 1))
  layer5:add(nn.ReLU())
  layer5:add(nn.Dropout(0.5))
  layer5:add(nn.SpatialConvolutionCuFFT(1024, 1024, 1, 1, 1, 1))
  layer5:add(nn.ReLU())
  layer5:add(nn.Dropout(0.5))
  layer5:add(nn.SpatialConvolutionCuFFT(1024, 10, 1, 1, 1, 1))
  layer5:add(nn.Reshape(10))
  layer5:add(nn.LogSoftMax())

  local pass5 = nn.ParallelTable()
  pass5:add(layer5)
  pass5:add(nn.LogSoftMax())
  pass5:add(nn.LogSoftMax())
  pass5:add(nn.LogSoftMax())

  model:add(pass5)
  local criterion = {nn.DistKLDivCriterion():cuda(), 
                     nn.DistKLDivCriterion():cuda(), 
                     nn.DistKLDivCriterion():cuda(), 
                     nn.DistKLDivCriterion():cuda()}
  return model:cuda(), criterion
end

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

config = {
  id = 'dsn1_val', 
  learningRateDecay = 1.0e-6, 
  momentum = 0.9, 
  batch_size = 128, 
  model_seed = 1, 
  train_jitter = true, 
  test_jitter = true, 
  early_stop = 12, 
  evaluate_every = 2, 
  table_output = true
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