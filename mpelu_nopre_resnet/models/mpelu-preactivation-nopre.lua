--  ************************************************************************
--  The implementation of MPELU nopre ResNet, arXiv:1606.00305 (https://arxiv.org/abs/1606.00305).
--  This code is modified from pre-ResNet (https://github.com/KaimingHe/resnet-1k-layers)
--  and fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  ************************************************************************

local nn = require 'nn'
require 'cunn'
require './mpelu/mpelu'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local alpha = 0.25
local beta  = 1

local function createModel(opt)
   local depth = opt.depth
   
   local function bottleneck(nInputPlane, nOutputPlane, stride)
      
      local nBottleneckPlane = nOutputPlane / 4
      
      if nInputPlane == nOutputPlane then -- most Residual Units have this shape      
         local convs = nn.Sequential()
         -- conv1x1
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(MPELU(alpha, beta, 5, 5, 10, 10, nBottleneckPlane))
         -- conv3x3
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(MPELU(alpha, beta, 5, 5, 10, 10, nBottleneckPlane))
         -- conv1x1
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Identity()
        
         return nn.Sequential()
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      else
         local block = nn.Sequential()     
         local convs = nn.Sequential()     
         -- conv1x1
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(MPELU(alpha, beta, 5, 5, 10, 10, nBottleneckPlane))
         -- conv3x3
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(MPELU(alpha, beta, 5, 5, 10, 10, nBottleneckPlane))
         -- conv1x1
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Sequential()
         shortcut:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
        
         return block
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      end
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
      local n = (depth - 2) / 9
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The new ResNet-164 and ResNet-1001 in [a]
	  local nStages = {16, 64, 128, 256}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1))
      model:add(SBatchNorm(nStages[1]))
      model:add(MPELU(alpha, beta, 5, 5, 10, 10, nStages[1]))
      model:add(layer(bottleneck, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(bottleneck, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(MPELU(alpha, beta, 5, 5, 10, 10, nStages[4]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
      local n = (depth - 2) / 9
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The new ResNet-164 and ResNet-1001 in [a]
	  local nStages = {16, 64, 128, 256}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1))
      model:add(SBatchNorm(nStages[1]))
      model:add(MPELU(alpha, beta, 5, 5, 10, 10, nStages[1]))
      model:add(layer(bottleneck, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(bottleneck, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(MPELU(alpha, beta, 5, 5, 10, 10, nStages[4]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n/(1 + alpha*alpha*beta*beta))) -- Taylor initialization for MPELU networks
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
