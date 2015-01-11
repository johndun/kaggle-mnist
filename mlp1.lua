require 'torch'
require 'nn'
require './lib/util'
require './lib/train'

config = {
  id = 'mlp1_val', 
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

local function crop_images(train_x)
  return batch_sample{src       = train_x, 
                      crp_off_x = 5, 
                      crp_off_y = 5, 
                      crp_sz_x  = 24,
                      crp_sz_y  = 24}
end

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)


print('## Early stopping using validation set')
local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
local test_x, test_y
local model, criterion = create_model(config.model_seed)
train_x = crop_images(train_x)
train_x, train_y, test_x, test_y = validation_split(train_x, train_y, 6144)
local preprocess_params = preprocess(train_x)
preprocess(test_x, preprocess_params)
local best_epoch = train(model, criterion, config, 
                         train_x, train_y, test_x, test_y)
print('Best model at epoch ' .. best_epoch)


print('\n## Train using full training set')
config.evalCounter = nil
config.epochs = best_epoch
config.eval = false
config.id = 'mlp1'
local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
local model, criterion = create_model(config.model_seed)
train_x = crop_images(train_x)
local preprocess_params = preprocess(train_x)
train(model, criterion, config, train_x, train_y)

print('\n## Generating predictions on the test set')
local test = torch.load(TEST_FNAME)
test = crop_images(test)
preprocess(test, preprocess_params)
local predictions = predict(model, test, config.batch_size)

print('\n## Writing predictions to file')
local file = io.open(string.format('result/%s_preds.csv', config.id), 'w')
file:write('ImageId,Label\n')
for i = 1, predictions:size(1) do
  file:write(string.format('%s,%s\n', i, CLASSES[predictions[i]]))
end