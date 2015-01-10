require 'torch'
require 'optim'
require 'xlua'
require 'cunn'

function test(model, test_x, test_y, batch_size)
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

function train(model, criterion, config, train_x, train_y, test_x, test_y)
  local parameters = model:getParameters()
  print('Number of model parameters: ' .. parameters:size(1))
  
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
      print('Accuracy on evaluation set:                       ' .. acc)
    end
    collectgarbage()
  end
end