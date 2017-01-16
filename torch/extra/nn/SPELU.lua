local SPELU, parent = torch.class('nn.SPELU','nn.Module')

function SPELU:__init(alpha, nOutputPlane)
   parent.__init(self)
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   local a = alpha or 1
   self.weight = torch.Tensor(nOutputPlane or 1):fill(a)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
end

function SPELU:updateOutput(input)
   input.THNN.SPELU_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.nOutputPlane
   )
   return self.output
end

function SPELU:updateGradInput(input, gradOutput)
   input.THNN.SPELU_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.nOutputPlane
   )
   return self.gradInput
end

function SPELU:accGradParameters(input, gradOutput, scale)
   self.gradWeightBuf = self.gradWeightBuf or input.new()
   self.gradWeightBuf2 = self.gradWeightBuf2 or input.new()
   input.THNN.SPELU_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.gradWeight:cdata(),
      self.gradWeightBuf:cdata(),
      self.gradWeightBuf2:cdata(),
      self.nOutputPlane,
      scale or 1
   )
   return self.gradWeight
end

function SPELU:clearState()
   nn.utils.clear(self, 'gradWeightBuf', 'gradWeightBuf2')
   return parent.clearState(self)
end
