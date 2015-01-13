require 'torch'
local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

require './convnet3.lua'
require './convnet3_1.lua'
require './convnet3_2.lua'