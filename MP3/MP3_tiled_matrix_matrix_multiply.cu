#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  
  //Define tileWidth variable
  const int tileWidth = 16;
  
  //Define shared memory tiles
  __shared__ float ds_A[tileWidth][tileWidth];
  __shared__ float ds_B[tileWidth][tileWidth];
   
  //Calculate the row index of the C and A
  int Row = blockIdx.y * tileWidth + threadIdx.y;
  //Calculate the column index of the C and B
  int Col = blockIdx.x * tileWidth + threadIdx.x;
  
  //Check whether numAColumns == numBRows, otherwise operation is invalid
  if (numAColumns == numBRows) {
    
    //Zero value of C cell
    float Cvalue = 0.0;
    
    //Define loop for each phase
    for (int phase = 0; phase < (numBRows -1)/tileWidth +1; ++phase) {
      
      //Check whether indices are valid for A tiles
      if ((Row < numARows) && (phase * tileWidth + threadIdx.x < numAColumns)){
        //Loading of A and B tiles into shared memory
        ds_A[threadIdx.y][threadIdx.x] = A[Row * numAColumns + (phase * tileWidth + threadIdx.x)];
      } else {
        ds_A[threadIdx.y][threadIdx.x] = 0.0;
      }
      
      //Check whether indices are valid for B tiles
      if ((phase * tileWidth + threadIdx.y < numBRows) && (Col < numBColumns)){
        //Loading of A and B tiles into shared memory
        ds_B[threadIdx.y][threadIdx.x] = B[(phase * tileWidth + threadIdx.y) * numBColumns + Col];
      } else {
        ds_B[threadIdx.y][threadIdx.x] = 0.0;
      }
      
      //Syncrhonize threads
      __syncthreads();
      
      //Loop per iteration 
      for (int iter = 0; iter < tileWidth; ++iter){
        Cvalue += ds_A[threadIdx.y][iter] * ds_B[iter][threadIdx.x];
      } /* end of iteration loop */
      
      //Synchronize threads
      __syncthreads();
      
    } /* end of phase loop */
    
  //Update C matrix
    if ((Row < numCRows) && (Col < numCColumns)){
      C[Row * numCColumns + Col] = Cvalue;
    }
  } /* end of valid operation check */
} /* end of matrixMultiplyShared */


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/16.0),ceil(numCRows/16.0),1);
  dim3 DimBlock(16,16,1);

  wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

