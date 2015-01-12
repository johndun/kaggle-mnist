require 'torch'
require 'optim'
require 'xlua'
require 'cunn'

local function sgd(model, criterion, config, train_x, train_y)
  local parameters, gradParameters = model:getParameters()
  local confusion = optim.ConfusionMatrix(CLASSES)
  config = config or {}
  local batch_size = config.batch_size or 12
  local shuffle = torch.randperm(train_x:size(1))
  local c = 1
  for t = 1, train_x:size(1), batch_size do
    if t + batch_size > train_x:size(1) then
      break
    end
    if opt.progress then
      xlua.progress(t, train_x:size(1))
    end
    local inputs = torch.Tensor(batch_size,
                                train_x:size(2),
                                train_x:size(3),
                                train_x:size(4))
    local targets = torch.Tensor(batch_size,
                                 train_y:size(2))
    for i = 1, batch_size do
      inputs[i]:copy(train_x[shuffle[t + i - 1]])
      targets[i]:copy(train_y[shuffle[t + i - 1]])
    end
    inputs = inputs:cuda()
    targets = targets:cuda()

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local output = model:forward(inputs)
      local f = criterion:forward(output, targets)
      local df_do = criterion:backward(output, targets)
      confusion:batchAdd(output, targets)      
      model:backward(inputs, df_do)
      return f, gradParameters
    end
    
    optim.sgd(feval, parameters, config)
    c = c + 1
    if c % 1000 == 0 then
      collectgarbage('collect')
    end
  end
  if opt.progress then
    xlua.progress(train_x:size(1), train_x:size(1))
  end
  confusion:updateValids()
  return confusion.totalValid
end

local function test(model, test_x, test_y, batch_size)
  local confusion = optim.ConfusionMatrix(CLASSES)
  local batch_size = batch_size or 12
  for t = 1, test_x:size(1), batch_size do
    if t + batch_size > test_x:size(1) then
      break
    end
    if opt.progress then
      xlua.progress(t, test_x:size(1))
    end
    local inputs = torch.Tensor(batch_size,
                                test_x:size(2),
                                test_x:size(3),
                                test_x:size(4))
    local targets = torch.Tensor(batch_size,
                                 test_y:size(2))
    for i = 1, batch_size do
      inputs[i]:copy(test_x[t + i - 1])
      targets[i]:copy(test_y[t + i - 1])
    end
    inputs = inputs:cuda()
    targets = targets:cuda()

    local output = model:forward(inputs)
    confusion:batchAdd(output, targets)
  end
  if opt.progress then
    xlua.progress(test_x:size(1), test_x:size(1))
  end
  confusion:updateValids()
  return confusion.totalValid
end

function predict(model, test_x, batch_size)
  local batch_size = batch_size or 12
  local preds = torch.Tensor(test_x:size(1))
  local inputs = torch.Tensor(batch_size,
                              test_x:size(2),
                              test_x:size(3),
                              test_x:size(4)):cuda()
  for t = 1, test_x:size(1), batch_size do
    if opt.progress then
      xlua.progress(t, test_x:size(1))
    end
    if t + batch_size > test_x:size(1) then
      batch_size = test_x:size(1) - t + 1
    end 
    inputs:narrow(1, 1, batch_size):copy(test_x:narrow(1, t, batch_size))
    local output = model:forward(inputs)
    local max_val, max_indices = output:max(2)
    preds:narrow(1, t, batch_size):copy(max_indices:narrow(1, 1, batch_size))
  end
  if opt.progress then
    xlua.progress(test_x:size(1), test_x:size(1))
  end
  return preds
end

function train(model, criterion, config, train_x, train_y, test_x, test_y)
  torch.manualSeed(config.train_seed)
  local parameters = model:getParameters()
  print('Number of model parameters: ' .. parameters:size(1))
  
  local best_test_acc = config.starting_acc or 0
  local best_epoch = 0
  for epoch = 1, config.epochs do
    model:training()
    print('\nTraining epoch ' .. epoch)
    local acc = sgd(model, criterion, config, train_x, train_y)
    print('Final learning rate: ' .. 
          (config.learningRate / 
          (1 + config.learningRateDecay * config.evalCounter)))
    print('Accuracy on training set: ' .. acc)
    if config.eval then
      model:evaluate()
      local acc = test(model, test_x, test_y, config.batch_size)
      local log_str = string.format(
        'Accuracy on evaluation set:                       %s', acc)
      if acc > best_test_acc then
        best_epoch = epoch
        best_test_acc = acc
        torch.save(string.format('model/%s.model', config.id), model)
        log_str = log_str .. '*'
      end
      print(log_str)
    end
    collectgarbage()
  end
  
  if config.eval then
    return best_epoch, best_test_acc
  else
    torch.save(string.format('model/%s.model', config.id), model)
  end
end