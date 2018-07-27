#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MPELU.cu"
#else

void THNN_(MPELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           long nOutputPlane)
{
  THCTensor_(resizeAs)(state, output, input);

  weight = THCTensor_(newContiguous)(state, weight);
  real *w = THCTensor_(data)(state, weight);
  bias = THCTensor_(newContiguous)(state, bias);
  real *b = THCTensor_(data)(state, bias);
  if (nOutputPlane == 0)
  {
    THC_pointwiseApply2(state, output, input, MPELUUpdateOutput<real>(w, b));
  }
  else
  {
    int ndim = THCTensor_(nDimension)(state, input);
    input = THCTensor_(newContiguous)(state, input);

    int n = THCTensor_(nElement)(state, input);
    if (input->size[ndim > 1] != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, input->size[ndim > 1]);

    int mapSize = 1;
    for (int d = 2; d < ndim; d++) {
      mapSize *= input->size[d];
    }
    int nElemsPerSample = nOutputPlane * mapSize;
    mpeluForward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, output),
      THCTensor_(data)(state, input),
      w, b, 
      n, nElemsPerSample, mapSize
    );
    THCudaCheck(cudaGetLastError());
    THCTensor_(free)(state, input);
  }

  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, bias);
}

void THNN_(MPELU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *bias,
           long nOutputPlane)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_check_nElement(state, output, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  weight = THCTensor_(newContiguous)(state, weight);
  real *w = THCTensor_(data)(state, weight);
  bias = THCTensor_(newContiguous)(state, bias);
  real *b = THCTensor_(data)(state, bias);
  if (nOutputPlane == 0)
  {
    THC_pointwiseApply3(state, gradInput, gradOutput, input, MPELUUpdateGradInput<real>(w, b));
  }
  else
  {
    int ndim = THCTensor_(nDimension)(state, input);
    input = THCTensor_(newContiguous)(state, input);
    output = THCTensor_(newContiguous)(state, output);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);

    int n = THCTensor_(nElement)(state, input);
    if (input->size[ndim > 1] != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, input->size[ndim > 1]);

    int mapSize = 1;
    for (int d = 2; d < ndim; d++) {
      mapSize *= input->size[d];
    }
    int nElemsPerSample = nOutputPlane * mapSize;
    mpeluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, gradInput),
      THCTensor_(data)(state, input),
      w, b,
      THCTensor_(data)(state, gradOutput),
      THCTensor_(data)(state, output),
      n, nElemsPerSample, mapSize
    );
    THCudaCheck(cudaGetLastError());
    THCTensor_(free)(state, input);
    THCTensor_(free)(state, output);
    THCTensor_(free)(state, gradOutput);
  }

  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, bias);
}

void THNN_(MPELU_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradWeight,
           THCTensor *gradWeightBuf,
           THCTensor *gradWeightBuf2,
           THCTensor *bias,
           THCTensor *gradBias,
           THCTensor *gradBiasBuf,
           THCTensor *gradBiasBuf2,
           long nOutputPlane,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_check_nElement(state, input, gradOutput);

  THCTensor *weighT = THCTensor_(newContiguous)(state, weight);
  real *w = THCTensor_(data)(state, weighT);
  THCTensor *biaS = THCTensor_(newContiguous)(state, bias);
  real *b = THCTensor_(data)(state, biaS);

  // use grad input for temporary storage, then call updateGradInput again
  if (nOutputPlane == 0)
  {
    THC_pointwiseApply3(state, gradInput, input, gradOutput, MPELUAccGradParametersShared<real>(w, b));

    // introduces a sync point
    real sum = ScalarConvert<accreal, real>::to(THCTensor_(sumall)(state, gradInput));
    real t = THCTensor_(get1d)(state, gradWeight, 0);
    THCTensor_(set1d)(state, gradWeight, 0, t + sum * scale);

    THC_pointwiseApply3(state, gradInput, input, gradOutput, MPELUAccGradParametersSharedForBeta<real>(w, b));

    // introduces a sync point
    sum = ScalarConvert<accreal, real>::to(THCTensor_(sumall)(state, gradInput));
    t = THCTensor_(get1d)(state, gradBias, 0);
    THCTensor_(set1d)(state, gradBias, 0, t + sum * scale);

    // restore gradInput
    THNN_(MPELU_updateGradInput)(state, input, output, gradOutput, gradInput, weight, bias, nOutputPlane);
  }
  else
  {
    int ndim = THCTensor_(nDimension)(state, input);

    if (ndim == 1)
    {
      THC_pointwiseApply3(state, gradWeight, input, gradOutput, MPELUAccGradParameters1to1<real>(scale, w, b));
      THC_pointwiseApply3(state, gradBias, input, gradOutput, MPELUAccGradParameters1to1ForBeta<real>(scale, w, b));
    }
    else
    {
    //   THCTensor gradInputP;
    //   THCTensor_(freeCopyTo)(state, gradInput, gradInputP);
      THCTensor *gradInputPartnar = THCTensor_(newContiguous)(state, gradInput);
      // THCTensor_(resizeAs)(state, gradInputPartnar, gradInput);
      // THCTensor_(resizeNd)(state, gradInputPartnar, input->nDimension, input->size, NULL);

      THC_pointwiseApply3(state, gradInput, input, gradOutput, MPELUAccGradParameters<real>(scale, w, b));
      THC_pointwiseApply3(state, gradInputPartnar, input, gradOutput, MPELUAccGradParametersForBeta<real>(scale, w, b));

      THCTensor *sumbuf = gradWeightBuf2;
      THCTensor_(resizeAs)(state, gradWeightBuf, gradWeight);

      THCTensor *sumbufBeta = gradBiasBuf2;
      THCTensor_(resizeAs)(state, gradBiasBuf, gradBias);

      if (ndim == 2)
      {
        THCTensor_(sum)(state, gradWeightBuf, gradInput, 0, 1);
        THCTensor_(cadd)(state, gradWeight, gradWeight, scale, gradWeightBuf);

        THCTensor_(sum)(state, gradBiasBuf, gradInputPartnar, 0, 1);
        THCTensor_(cadd)(state, gradBias, gradBias, scale, gradBiasBuf);
      }
      else
      {
        THCTensor *buffer = THCTensor_(newContiguous)(state, gradInput);
        THCTensor *bufferBeta = THCTensor_(newContiguous)(state, gradInputPartnar);
        long size3 = 1;
        for (int d = 2; d < ndim; d++) {
          size3 *= input->size[d];
        }
        THCTensor_(resize3d)(state, buffer, input->size[0], nOutputPlane, size3);
        THCTensor_(resize2d)(state, sumbuf, input->size[0], nOutputPlane);
        THCTensor_(sum)(state, sumbuf, buffer, 2, 1);
        THCTensor_(sum)(state, gradWeightBuf, sumbuf, 0, 1);
        THCTensor_(cadd)(state, gradWeight, gradWeight, scale, gradWeightBuf);
        THCTensor_(free)(state, buffer);

        THCTensor_(resize3d)(state, bufferBeta, input->size[0], nOutputPlane, size3);
        THCTensor_(resize2d)(state, sumbufBeta, input->size[0], nOutputPlane);
        THCTensor_(sum)(state, sumbufBeta, bufferBeta, 2, 1);
        THCTensor_(sum)(state, gradBiasBuf, sumbufBeta, 0, 1);
        THCTensor_(cadd)(state, gradBias, gradBias, scale, gradBiasBuf);
        THCTensor_(free)(state, bufferBeta);
      }
      THCTensor_(free)(state, gradInputPartnar);
      // restore gradInput
      THNN_(MPELU_updateGradInput)(state, input, output, gradOutput, gradInput, weight, bias, nOutputPlane);
    }
  }
}

#endif
