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


__global__ void singlePassScan(float *input, float *output, int len, int pass) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  __shared__ float sharedArray[BLOCK_SIZE * 2];
  int loadIndex;
  int loadStride;
  if (pass == 1) {
    loadIndex = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    loadStride = blockDim.x;
  } else if (pass == 2) {
    loadIndex = (threadIdx.x + 1) * blockDim.x * 2 - 1;
    loadStride = 2 * blockDim.x;
  } else {
    return;
  }

  if(loadIndex < len){
        sharedArray[threadIdx.x] = input[loadIndex];
  } else {
        sharedArray[threadIdx.x] = 0; 
  }

  if(loadIndex + loadStride < len){
        sharedArray[threadIdx.x + blockDim.x] = input[loadIndex + loadStride];
  } else {
        sharedArray[threadIdx.x + blockDim.x] = 0;
  }
  __syncthreads();

  
  // reduction phase
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
    if (localIdx < 2 * blockDim.x) {
      sharedArray[localIdx] += sharedArray[localIdx - stride];
    }
  }

  // reverse phase
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int localIdx = (threadIdx.x + 1) * stride * 2 - 1;
    if (localIdx + stride < 2 * blockDim.x) {
      sharedArray[localIdx + stride] += sharedArray[localIdx];
    }
  }

  // store partial results to output
  __syncthreads();
  if (2 * blockIdx.x * blockDim.x + threadIdx.x < len) {
    output[2 * blockIdx.x * blockDim.x + threadIdx.x] = sharedArray[threadIdx.x];
  }
  if (2 * blockIdx.x * blockDim.x + threadIdx.x + blockDim.x < len) {
    output[2 * blockIdx.x * blockDim.x + threadIdx.x + blockDim.x] = sharedArray[threadIdx.x + blockDim.x];
  }
}

__global__ void scanSum(float *input, float *output, float *sum, int len) {
  __shared__ float increment;
  int localIndex = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      increment = 0;
    } else {
      increment = sum[blockIdx.x - 1];
    }
  }
  __syncthreads();

  if (localIndex < len) {
    output[localIndex] = input[localIndex] + increment;
  }
  if (localIndex + blockDim.x) {
    output[localIndex + blockDim.x] = input[localIndex + blockDim.x] + increment;
  }
}


// void recursiveScan (float *input, float *scan_buffer, float *scan_sums, float *output, int len){
  
//   // FIRST PASS

//   //@@ Initialize the grid and block dimensions here
//   dim3 dimGrid(ceil(len/float(BLOCK_SIZE * 2)), 1, 1);
//   dim3 dimBlock(BLOCK_SIZE, 1, 1);
//   dim3 singleGrid(1, 1, 1);

//   int firstloadIndex = 2 * blockIdx.x * blockDim.x + threadIdx.x;
//   int firstLoadStride = blockDim.x;
//   singlePassScan<<<dimGrid, dimBlock>>>(input, scan_buffer, len, firstloadIndex, firstLoadStride);

//   // SECOND PASS
//   int secondloadIndex = (threadIdx.x + 1) * blockDim.x * 2 - 1;
//   int secondLoadStride = 2 * blockDim.x;
//   singlePassScan<<<singleGrid, dimBlock>>>(scan_buffer, scan_sums, len, secondloadIndex, secondLoadStride);

//   // SUM
//   scanSum<<<dimGrid, dimBlock>>>(scan_buffer, output, scan_sums, len, firstloadIndex);
// }



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceTempScan; // Temporary buffer for scan
  float *deviceTempSums;   // Temporary buffer for scan sums
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
  wbCheck(cudaMalloc((void **)&deviceTempScan, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceTempSums, 2 * BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/float(BLOCK_SIZE * 2)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 singleGrid(1, 1, 1);

  // FIRST PASS
  singlePassScan<<<dimGrid, dimBlock>>>(deviceInput, deviceTempScan, numElements, 1);
  cudaDeviceSynchronize();

  // SECOND PASS
  singlePassScan<<<singleGrid, dimBlock>>>(deviceTempScan, deviceTempSums, numElements, 2);
  cudaDeviceSynchronize();

  // SUM
  scanSum<<<dimGrid, dimBlock>>>(deviceTempScan, deviceOutput, deviceTempSums, numElements);
  cudaDeviceSynchronize();

  // recursiveScan(deviceInput, deviceTempScan, deviceTempSums, deviceOutput, numElements);


  // cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceTempScan);
  cudaFree(deviceTempSums);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}