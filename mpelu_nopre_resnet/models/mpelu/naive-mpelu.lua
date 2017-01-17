-- naive implementation of MPELU
require 'nn'
require 'nnlr'

function MPELU(alpha, beta, lr_a, lr_b, reg_a, reg_b, nOutputPlane)
    local mpelu = nn.Sequential()
    mpelu:add(nn.PReLU(beta, nOutputPlane)
                      :learningRate('weight', lr_a)
                      :weightDecay('weight', reg_a)
    )
    mpelu:add(nn.ELU(1, true))
    mpelu:add(nn.PReLU(alpha, nOutputPlane)
                      :learningRate('weight', lr_b)
                      :weightDecay('weight', reg_b)  
    )

    return mpelu
end

return MPELU
