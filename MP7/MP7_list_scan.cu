// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void reductionPhaseKernel() {
  // XY[22 * BLOCK_SIZE] from shared memory
  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
    if (localIdx < 2 * BLOCK_SIZE) {
      sharedArray[localIdx] += sharedArray[localIdx - stride];
    __syncthreads();
    }
  }
}

__global__ void postReductionReversePhase() {
  for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
    if (localIdx < 2 * BLOCK_SIZE) {
      XY[localIdx + stride] += XY[localIdx];
    }
  }
  __syncthreads();
  if (i < InputSize) {
    Output[i] = sharedArray[threadIdx.x];
  }
}


__global__ void recursiveScan(float *input, float *output, int len, int level) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sharedArray[BLOCK_SIZE * 2];

  if(2 * index + BLOCK_SIZE * blockIdx.x < len){
        sharedArray[2 * threadIdx.x] = input[2 * index + BLOCK_SIZE * blockIdx.x];
  } else {
        sharedArray[2 * threadIdx.x] = 0; 
  }
   if(2 * index + BLOCK_SIZE * blockIdx.x + 1 < len){
        sharedArray[2 * threadIdx.x + 1] = input[2 * index + BLOCK_SIZE * blockIdx.x + 1];
  } else {
        sharedArray[2 * threadIdx.x + 1] = 0;
  }
  __syncthreads();

  int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
  
  // reduction phase
  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    if (localIdx < 2 * BLOCK_SIZE) {
      sharedArray[localIdx] += sharedArray[localIdx - stride];
    __syncthreads();
    }
  }

  // reverse phase
  for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    // int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
    if (localIdx < 2 * BLOCK_SIZE) {
      sharedArray[localIdx + stride] += sharedArray[localIdx];
      __syncthreadS();
    }
  }

  // store partial results to output
  if (blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x < len) {
    output[blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x] = sharedArray[threadIdx.x];
  }
  if (blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x + blockDim.x < len) {
    output[blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x + blockDim.x] = sharedArray[threadIdx.x + blockDim.x];
  }


}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce



  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

