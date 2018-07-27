local MPELU, parent = torch.class('nn.MPELU','nn.Module')

function MPELU:__init(alpha, beta, nOutputPlane)
   parent.__init(self)
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   local a = alpha or 1
   local b = beta or 1
   self.weight = torch.Tensor(nOutputPlane or 1):fill(a)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
   self.bias = torch.Tensor(nOutputPlane or 1):fill(b)
   self.gradBias = torch.Tensor(nOutputPlane or 1)
end

function MPELU:updateOutput(input)
   input.THNN.MPELU_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.nOutputPlane
   )
   return self.output
end

function MPELU:updateGradInput(input, gradOutput)
   input.THNN.MPELU_updateGradInput(
      input:cdata(),
      self.output:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.nOutputPlane
   )
   return self.gradInput
end

function MPELU:accGradParameters(input, gradOutput, scale)
   self.gradWeightBuf = self.gradWeightBuf or input.new()
   self.gradWeightBuf2 = self.gradWeightBuf2 or input.new()
   self.gradBiasBuf = self.gradBiasBuf or input.new()
   self.gradBiasBuf2 = self.gradBiasBuf2 or input.new()
   input.THNN.MPELU_accGradParameters(
      input:cdata(),
      self.output:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.gradWeight:cdata(),
      self.gradWeightBuf:cdata(),
      self.gradWeightBuf2:cdata(),
      self.bias:cdata(),
      self.gradBias:cdata(),
      self.gradBiasBuf:cdata(),
      self.gradBiasBuf2:cdata(),
      self.nOutputPlane,
      scale or 1
   )
   return self.gradWeight, self.gradBias
end

function MPELU:clearState()
   nn.utils.clear(self, 'gradWeightBuf', 'gradWeightBuf2')
   nn.utils.clear(self, 'gradBiasBuf', 'gradBiasBuf2')
   return parent.clearState(self)
end
