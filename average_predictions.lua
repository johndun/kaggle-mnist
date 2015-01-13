require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
require 'fbcunn'
FB = true

local val_sz = 6144
config = {batch_size = 128}

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

function average_predictions(test_x, ids, jitters)
  local function get_predictions(test, id, jitter)
    print('\n## Generating predictions on the test set')
    config.id = id
    config.test_jitter = jitter
    
    local test = test
    if not jitter then
      test = crop_images(test)
    end
    local model = torch.load(string.format('model/%s.model', config.id))
    local preprocess_params = torch.load(string.format(
      'model/%s_preproc_params.t7', config.id))
    preprocess(test, preprocess_params)
    local probs = predict(model, test, config.batch_size)
    return probs
  end
  
  local probs = torch.Tensor(test_x:size(1), #CLASSES):zero()
  for i = 1, #ids do
    probs:add(get_predictions(test_x, ids[i], jitters[i]))
    collectgarbage('collect')
  end
  return probs
end

-- local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
-- local test_x, test_y
-- train_x, train_y, test_x, test_y = validation_split(train_x, train_y, val_sz)
-- local jitters = {true, true, true}
-- local ids = {'convnet3_val', 'convnet3_1_val', 'convnet3_2_val'}
-- local probs = average_predictions(test_x, ids, jitters)
-- local confusion = optim.ConfusionMatrix(CLASSES)
-- confusion:batchAdd(probs, test_y)
-- confusion:updateValids()
-- print(confusion)

local test_x = torch.load(TEST_FNAME)
local jitters = {true, true, true}
local ids = {'convnet3', 'convnet3_1', 'convnet3_2'}
local probs = average_predictions(test_x, ids, jitters)
local max_vals, max_indices = probs:max(2)
max_indices = max_indices:reshape(max_indices:size(1))

print('\n## Writing predictions to file')
local file = io.open(string.format('result/ensemble1_preds.csv', 
                                   config.id), 'w')
file:write('ImageId,Label\n')
for i = 1, max_indices:size(1) do
  file:write(string.format('%s,%s\n', i, CLASSES[max_indices[i]]))
end