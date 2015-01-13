require 'torch'
require 'csvigo'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

CLASSES = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
TRAIN_FNAME = 'data/train.t7'
TEST_FNAME = 'data/test.t7'

-- train
local dat = csvigo.load{path='data/train.csv', mode='raw'}
local n_samples = #dat - 1
local x0 = torch.Tensor(n_samples, 1, 28, 28)
local dat_y = torch.Tensor(n_samples, #CLASSES):zero()
for i = 1, n_samples do
  local y = table.remove(dat[i + 1], 1)
  local x = torch.Tensor(dat[i + 1]):reshape(1, 28, 28)
  x0[i]:copy(x)
  dat_y[i][y + 1] = 1
end
local dat_x = torch.Tensor(n_samples, 1, 32, 32)
dat_x:narrow(3, 3, 28):narrow(4, 3, 28):copy(x0)
torch.save(TRAIN_FNAME, {dat_x, dat_y})

-- test
local dat = csvigo.load{path='data/test.csv', mode='raw'}
local n_samples = #dat - 1
local dat_x = torch.Tensor(n_samples, 1, 32, 32)
for i = 1, n_samples do
  local x = torch.Tensor(dat[i + 1]):reshape(1, 28, 28)
  dat_x[i]:narrow(2, 3, 28):narrow(3, 3, 28):copy(x)
end
torch.save(TEST_FNAME, dat_x)

local train0_fname = 'data/train_32x32.t7'
local train_fname = 'data/train_yl.t7'
local test0_fname = 'data/test_32x32.t7'
local test_fname = 'data/test_yl.t7'
local function load_data(fname)
    local f = torch.load(fname,'ascii')
    local dat = f.data:type(torch.getdefaulttensortype())
    local labs = f.labels
    local labels = torch.Tensor(labs:size(1), 10):zero()
    for i = 1, labs:size(1) do
      labels[i][labs[i]] = 1.0
    end
    return dat, labels
end
local train_x,train_y = load_data(train0_fname)
local test_x,test_y = load_data(test0_fname)
torch.save(train_fname, {train_x,train_y})
torch.save(test_fname, {test_x,test_y})

-- require 'image'
-- local image_tile = dat_x:narrow(1,1,100)
-- image_tile = image.toDisplayTensor{input=image_tile, padding=4, nrow=10}
-- image.saveJPG('deleteme.jpg', image_tile)