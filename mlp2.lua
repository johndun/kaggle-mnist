require 'torch'
require 'nn'
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

config = {
  id = 'mlp2_val', 
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
config.id = 'mlp2'
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
  train(model, criterion, config, train_x, train_y)
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