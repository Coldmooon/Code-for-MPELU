#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "common.h"

template <typename T>
struct SPELUUpdateOutput
{
  T* weight_;

  SPELUUpdateOutput(T* weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : weight_[0] * (exp(x) - 1);
  }
};

template <typename T>
__global__ void speluForward(T *output, const T *input, const T *weight, int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    output[i] = input[i] > 0 ? input[i] : (exp(input[i]) - 1) * weight[mapNumber];
  }
}

template <typename T>
struct SPELUUpdateGradInput
{
  T *weight_;

  SPELUUpdateGradInput(T *weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *gradOutput, T *input)
  {
    *gradInput = *input > 0 ? *gradOutput : *gradOutput * *weight_ * exp(*input);
  }
};

template <typename T>
__global__ void speluBackward(
  T *gradInput,
  const T *input,
  const T *weight,
  const T *gradOutput,
  int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * weight[mapNumber] * exp(input[i]);
  }
}

template <typename T>
struct SPELUAccGradParametersShared
{
  __device__ __forceinline__ void operator()(T *gradInput, T  *input, T *gradOutput)
  {
    *gradInput = (exp(*input) - 1) * (*gradOutput) * (*input <= 0);
  }
};

template <typename T>
struct SPELUAccGradParameters
{
  T scale;

  SPELUAccGradParameters(T scale)
    : scale(scale)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput)
  {
    *gradInput = (exp(*input) - 1) * (*gradOutput) * scale * (*input <= 0);
  }
};

template <typename T>
struct SPELUAccGradParameters1to1
{
  T scale;

  SPELUAccGradParameters1to1(T scale)
    : scale(scale)
  {}

  __device__ __forceinline__ void operator()(T *gradWeight, T *input, T *gradOutput)
  {
    *gradWeight += (exp(*input) - 1) * (*gradOutput) * scale * (*input <= 0);
  }
};

#include "generic/SPELU.cu"
#include "THCGenerateFloatTypes.h"
