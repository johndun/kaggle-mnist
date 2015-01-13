require 'torch'
require 'optim'
require 'xlua'
require 'cunn'

local function sgd(model, criterion, config, train_x, train_y)
  local parameters, gradParameters = model:getParameters()
  local confusion = optim.ConfusionMatrix(CLASSES)
  local batch_size = config.batch_size or 12
  local shuffle = torch.randperm(train_x:size(1))
  local c = 1
  for t = 1, train_x:size(1), batch_size do
    if t + batch_size - 1 > train_x:size(1) then
      break
    end
    if opt.progress then
      xlua.progress(t, train_x:size(1))
    end
    local inputs = torch.Tensor(batch_size,
                                INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3])
    local targets = torch.Tensor(batch_size, #CLASSES)
    for i = 1, batch_size do
      local img = train_x[shuffle[t + i - 1]]
      if config.train_jitter then
        img = random_jitter(img)
      end
      inputs[i]:copy(img)
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

local function test_augmented(model, test_x, test_y, batch_size)
  local confusion = optim.ConfusionMatrix(CLASSES)
  local batch_size = batch_size or 12
  local inputs = torch.Tensor(batch_size,
                              INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3]):cuda()
  batch_size = math.floor(batch_size / TEST_JITTER_SZ)
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local test_n = test_x:size(1)
  for t = 1, test_n, batch_size do
    if opt.progress then
      xlua.progress(t, test_n)
    end
    if t + batch_size - 1 > test_n then
      batch_size = test_n - t + 1
    end
    for i = 1, batch_size do
      local jittered_x = test_jitter(test_x[t + i - 1])
      inputs:narrow(1, 
                    1 + TEST_JITTER_SZ*(i-1), 
                    TEST_JITTER_SZ):copy(jittered_x)
      targets[i]:copy(test_y[t + i - 1])
    end
    local output = model:forward(inputs)
    for i = 1, batch_size do
      local preds = output:narrow(1, 
                    1 + TEST_JITTER_SZ*(i-1), 
                    TEST_JITTER_SZ):mean(1):reshape(#CLASSES)
      confusion:add(preds, targets[i])
    end
  end
  if opt.progress then
    xlua.progress(test_n, test_n)
  end
  confusion:updateValids()
  return confusion.totalValid
end

local function test(model, test_x, test_y, batch_size)
  if config.test_jitter then
    return test_augmented(model, test_x, test_y, batch_size)
  end
  
  local confusion = optim.ConfusionMatrix(CLASSES)
  local batch_size = batch_size or 12
  local inputs = torch.Tensor(batch_size,
                              INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3]):cuda()
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local test_n = test_x:size(1)
  for t = 1, test_n, batch_size do
    if opt.progress then
      xlua.progress(t, test_n)
    end
    if t + batch_size - 1 > test_n then
      batch_size = test_n - t + 1
    end
    for i = 1, batch_size do
      inputs[i]:copy(test_x[t + i - 1])
      targets[i]:copy(test_y[t + i - 1])
    end

    local output = model:forward(inputs)
    confusion:batchAdd(output:narrow(1, 1, batch_size), 
                       targets:narrow(1, 1, batch_size))
  end
  if opt.progress then
    xlua.progress(test_n, test_n)
  end
  confusion:updateValids()
  return confusion.totalValid
end

local function training_loop(model, criterion, config, 
                             train_x, train_y, test_x, test_y)
  torch.manualSeed(config.train_seed)
  local parameters = model:getParameters()
  print('Number of model parameters: ' .. parameters:size(1))
  
  local best_test_acc = config.starting_acc or 0
  local evaluate_every = config.evaluate_every or 1
  local best_epoch = 0
  local epochs_since_best = 0
  for epoch = 1, config.epochs do
    model:training()
    print('\nTraining epoch ' .. epoch)
    local acc = sgd(model, criterion, config, train_x, train_y)
    print('Final learning rate: ' .. 
          (config.learningRate / 
          (1 + config.learningRateDecay * config.evalCounter)))
    print('Accuracy on training set: ' .. acc)
    if config.eval and epoch % evaluate_every == 0 then
      model:evaluate()
      local acc = test(model, test_x, test_y, config.batch_size)
      local log_str = string.format(
        'Accuracy on evaluation set:                       %s', acc)
      if acc > best_test_acc then
        best_epoch = epoch
        best_test_acc = acc
        torch.save(string.format('model/%s.model', config.id), model)
        log_str = log_str .. '*'
        epochs_since_best = 0
      else
        epochs_since_best = epochs_since_best + evaluate_every
        if config.early_stop and epochs_since_best >= config.early_stop then
          print(log_str)
          print(string.format('Stopping, no improvement in %s epochs', 
                              config.early_stop))
          break
        end
      end
      print(log_str)
    end
    collectgarbage()
  end
  
  if config.eval then
    return best_epoch, best_test_acc
  else
    torch.save(string.format('model/%s.model', config.id), model)
    return model
  end
end

local function predict_augmented(model, test_x, batch_size)
  local batch_size = batch_size or 12
  local preds = torch.Tensor(test_x:size(1), #CLASSES)
  local inputs = torch.Tensor(batch_size,
                              INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3]):cuda()
  batch_size = math.floor(batch_size / TEST_JITTER_SZ)
  local test_n = test_x:size(1)
  for t = 1, test_n, batch_size do
    if opt.progress then
      xlua.progress(t, test_n)
    end
    if t + batch_size - 1 > test_n then
      batch_size = test_n - t + 1
    end 
    for i = 1, batch_size do
      local jittered_x = test_jitter(test_x[t + i - 1])
      inputs:narrow(1, 
                    1 + TEST_JITTER_SZ*(i-1), 
                    TEST_JITTER_SZ):copy(jittered_x)
    end
    local output = model:forward(inputs)
    for i = 1, batch_size do
      local out = output:narrow(1, 
                                1 + TEST_JITTER_SZ*(i-1), 
                                TEST_JITTER_SZ):mean(1):reshape(#CLASSES)
      preds[t + i - 1]:copy(out)
    end
  end
  if opt.progress then
    xlua.progress(test_n, test_n)
  end
  return preds
end

function predict(model, test_x, batch_size)
  if config.test_jitter then
    return predict_augmented(model, test_x, batch_size)
  end
  local batch_size = batch_size or 12
  local preds = torch.Tensor(test_x:size(1), #CLASSES)
  local inputs = torch.Tensor(batch_size,
                              INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3]):cuda()
  for t = 1, test_x:size(1), batch_size do
    if opt.progress then
      xlua.progress(t, test_x:size(1))
    end
    if t + batch_size > test_x:size(1) then
      batch_size = test_x:size(1) - t + 1
    end 
    inputs:narrow(1, 1, batch_size):copy(test_x:narrow(1, t, batch_size))
    local output = model:forward(inputs)
    preds:narrow(1, t, batch_size):copy(output:narrow(1, 1, batch_size))
  end
  if opt.progress then
    xlua.progress(test_x:size(1), test_x:size(1))
  end
  return preds
end

function gen_predictions(model)
  print('\n## Generating predictions on the test set')
  local test = torch.load(TEST_FNAME)
  if not config.test_jitter then
    test = crop_images(test)
  end
  local preprocess_params = torch.load(string.format(
    'model/%s_preproc_params.t7', config.id))
  preprocess(test, preprocess_params)
  local probs = predict(model, test, config.batch_size)
  local max_vals, max_indices = probs:max(2)
  max_indices = max_indices:reshape(max_indices:size(1))
  
  print('\n## Writing predictions to file')
  local file = io.open(string.format('result/%s_preds.csv', config.id), 'w')
  file:write('ImageId,Label\n')
  for i = 1, max_indices:size(1) do
    file:write(string.format('%s,%s\n', i, CLASSES[max_indices[i]]))
  end
end

function validate(model, criterion, learning_rates, seeds, epochs, val_sz)
  print('### Early stopping using validation set')
  local epochs = epochs
  config.eval = true
  local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
  local test_x, test_y, preprocess_params
  train_x, train_y, test_x, test_y = validation_split(train_x, train_y, val_sz)
  if not config.train_jitter then
    train_x = crop_images(train_x)
    test_x = crop_images(test_x)
    preprocess_params = preprocess(train_x)
  else
    local tmp = batch_sample{src       = train_x, 
                             crp_off_x = 3, 
                             crp_off_y = 3, 
                             crp_sz_x  = 28,
                             crp_sz_y  = 28, 
                             out_w     = INPUT_SZ[3], 
                             out_h     = INPUT_SZ[2]}
    preprocess_params = preprocess(tmp)
    preprocess(train_x, preprocess_params)
  end
  preprocess(test_x, preprocess_params)
  
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning rate %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    epochs[i], config.starting_acc = training_loop(model, criterion, config, 
                                                   train_x, train_y, 
                                                   test_x, test_y)
    model = torch.load(string.format('model/%s.model', config.id))
  end
  return epochs
end

function train(model, criterion, learning_rates, seeds, epochs)
  print('\n### Train using full training set')
  config.eval = false
  local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
  local preprocess_params
  if not config.train_jitter then
    train_x = crop_images(train_x)
    preprocess_params = preprocess(train_x)
  else
    local tmp = batch_sample{src       = train_x, 
                             crp_off_x = 3, 
                             crp_off_y = 3, 
                             crp_sz_x  = 28,
                             crp_sz_y  = 28, 
                             out_w     = INPUT_SZ[3], 
                             out_h     = INPUT_SZ[2]}
    preprocess_params = preprocess(tmp)
    preprocess(train_x, preprocess_params)
  end
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    if config.epochs > 0 then
      model = training_loop(model, criterion, config, train_x, train_y)
    end
  end
  return model
end