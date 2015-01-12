require 'torch'
require 'nn'
require 'fbcunn'
require './lib/util'
require './lib/train'

local function crop_images(train_x)
  return batch_sample{src       = train_x, 
                      crp_off_x = 5, 
                      crp_off_y = 5, 
                      crp_sz_x  = 24,
                      crp_sz_y  = 24}
end

local function create_model()
  local model = nn.Sequential() 
  model:add(nn.SpatialConvolutionCuFFT(1, 128, 5, 5, 1, 1))
  -- model:add(nn.SpatialConvolutionMM(1, 128, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolutionCuFFT(128, 256, 5, 5, 1, 1))
  -- model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionCuFFT(256, 512, 4, 4, 1, 1))
  -- model:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionCuFFT(512, 1024, 2, 2, 1, 1))
  -- model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionCuFFT(1024, #CLASSES, 1, 1, 1, 1))
  -- model:add(nn.SpatialConvolutionMM(1024, #CLASSES, 1, 1, 1, 1))
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
  epochs = 20, 
  model_seed = 1, 
  eval = true, 
  jitter = false
}

print('### Early stopping using validation set')
local learning_rates = {1.0, 0.1}
local seeds = {2, 3}
local epochs = {20, 10}

local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
local test_x, test_y
train_x = crop_images(train_x)
train_x, train_y, test_x, test_y = validation_split(train_x, train_y, 6144)
local preprocess_params = preprocess(train_x)
preprocess(test_x, preprocess_params)
local model, criterion = create_model(config.model_seed)

for i = 1, #learning_rates do
  print(string.format('\n### Training at learning rate %s: %s', 
                      i, learning_rates[i]))
  config.learningRate = learning_rates[i]
  config.train_seed = seeds[i]
  config.epochs = epochs[i]
  config.evalCounter = nil
  epochs[i], config.starting_acc = train(model, criterion, config, 
                                         train_x, train_y, test_x, test_y)
  model = torch.load(string.format('model/%s.model', config.id))
end

print('\n### Train using full training set')
config.eval = false
config.id = string.gsub(config.id, '_val$', '')
local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
train_x = crop_images(train_x)
local preprocess_params = preprocess(train_x)
local model, criterion = create_model(config.model_seed)
for i = 1, #learning_rates do
  print(string.format('\n### Training at learning %s: %s', 
                      i, learning_rates[i]))
  config.learningRate = learning_rates[i]
  config.train_seed = seeds[i]
  config.epochs = epochs[i]
  config.evalCounter = nil
  if config.epochs > 0 then
    train(model, criterion, config, train_x, train_y)
  end
end

print('\n### Generating predictions on the test set')
local test = torch.load(TEST_FNAME)
test = crop_images(test)
preprocess(test, preprocess_params)
local predictions = predict(model, test, config.batch_size)

print('\n### Writing predictions to file')
local file = io.open(string.format('result/%s_preds.csv', config.id), 'w')
file:write('ImageId,Label\n')
for i = 1, predictions:size(1) do
  file:write(string.format('%s,%s\n', i, CLASSES[predictions[i]]))
end