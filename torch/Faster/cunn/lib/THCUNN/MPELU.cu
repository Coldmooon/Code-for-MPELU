#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

#include "common.h"

template <typename T>
struct MPELUUpdateOutput
{
  T* weight_;
  T* bias_;
  MPELUUpdateOutput(T* weight, T* bias)
    : weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : weight_[0] * (exp(bias_[0] * x) - 1);
  }
};

template <typename T>
__global__ void mpeluForward(T *output, const T *input, const T *weight, const T *bias, int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    output[i] = input[i] > 0 ? input[i] : (exp(bias[mapNumber] * input[i]) - 1) * weight[mapNumber];
  }
}

template <typename T>
struct MPELUUpdateGradInput
{
  T *weight_;
  T *bias_;
  MPELUUpdateGradInput(T *weight, T *bias)
    : weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *gradOutput, T *input)
  {
    *gradInput = *input > 0 ? *gradOutput : *gradOutput * ( *weight_ * *bias_ * exp(*bias_ * *input) );
  }
};

template <typename T>
__global__ void mpeluBackward(
  T *gradInput,
  const T *input,
  const T *weight, const T *bias,
  const T *gradOutput, const T *output,
  int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    // gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * weight[mapNumber] * bias[mapNumber] * exp(bias[mapNumber] * input[i]);
    gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * bias[mapNumber] * (output[i] + weight[mapNumber]);
  }
}

template <typename T>
struct MPELUAccGradParametersShared
{
  T *weight_;
  T *bias_;
  MPELUAccGradParametersShared(T *weight, T *bias)
    : weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T  *input, T *gradOutput)
  {
    *gradInput = (exp(*bias_ * *input) - 1) * (*gradOutput) * (*input <= 0);
  }
};

template <typename T>
struct MPELUAccGradParametersSharedForBeta
{
  T *weight_;
  T *bias_;
  MPELUAccGradParametersSharedForBeta(T *weight, T *bias)
    : weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T  *input, T *gradOutput)
  {
    *gradInput = *weight_ * exp(*bias_ * *input) * *input * (*gradOutput) * (*input <= 0);
  }
};

template <typename T>
struct MPELUAccGradParameters
{
  T scale;
  T *weight_;
  T *bias_;
  MPELUAccGradParameters(T scale, T *weight, T *bias)
    : scale(scale), weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput)
  {
    *gradInput = (exp(*bias_ * *input) - 1) * (*gradOutput) * scale * (*input <= 0);
  }
};

template <typename T>
struct MPELUAccGradParametersForBeta
{
  T scale;
  T *weight_;
  T *bias_;
  MPELUAccGradParametersForBeta(T scale, T *weight, T *bias)
    : scale(scale), weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput)
  {
    *gradInput = *weight_ * exp(*bias_ * *input) * *input * (*gradOutput) * scale * (*input <= 0);
  }
};

template <typename T>
struct MPELUAccGradParameters1to1
{
  T scale;
  T *weight_;
  T *bias_;
  MPELUAccGradParameters1to1(T scale, T *weight, T *bias)
    : scale(scale), weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradWeight, T *input, T *gradOutput)
  {
    *gradWeight += (exp(*bias_ * *input) - 1) * (*gradOutput) * scale * (*input <= 0);
  }
};

template <typename T>
struct MPELUAccGradParameters1to1ForBeta
{
  T scale;
  T *weight_;
  T *bias_;
  MPELUAccGradParameters1to1ForBeta(T scale, T *weight, T *bias)
    : scale(scale), weight_(weight), bias_(bias)
  {}

  __device__ __forceinline__ void operator()(T *gradBias, T *input, T *gradOutput)
  {
    *gradBias += *weight_ * exp(*bias_ * *input) * *input * (*gradOutput) * scale * (*input <= 0);
  }
};

#include "generic/MPELU.cu"
#include "THCGenerateFloatTypes.h"
