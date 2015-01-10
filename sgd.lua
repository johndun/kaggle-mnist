require 'torch'
require 'optim'
require 'xlua'

function sgd(model, criterion, config, train_x, train_y)
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
