#include <wb.h>

#define wbCheck(stmt)                                                     \
do {                                                                      \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                           \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));        \
    return -1;                                                            \
    }                                                                     \
} while (0)

#define maskWidth 5
#define maskRadius maskWidth / 2

//@@ INSERT CODE HERE
#define tileWidth 12
#define blockWidth tileWidth + maskWidth -1

// Allocate space for mask in constant memory
__constant__ float maskConst[maskWidth][maskWidth];

// Define 2D convolutional kernel
__global__ void convKernelShared(float *inputImage, float *outputImage,
                                 int curChannel, int imgHeight,
                                 int imgWidth, int imgChannels){
    
    // Define shared memory tile
    __shared__ float sharedTile[tileWidth + maskWidth - 1][tileWidth + maskWidth - 1];
    
    int outputCol = blockIdx.x * tileWidth + threadIdx.x;
    int outputRow = blockIdx.y * tileWidth + threadIdx.y;

    int inputCol = outputCol - maskRadius;
    int inputRow = outputRow - maskRadius;
    
    // Load data into shared memory
    if (inputCol >= 0 && inputCol < imgWidth && inputRow >= 0 && inputRow < imgHeight) {
        sharedTile[threadIdx.y][threadIdx.x] = inputImage[(inputRow * imgWidth + inputCol) * imgChannels + curChannel];
    } else {
        sharedTile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Compute 2D image convolution
    float outputValue = 0.0f;
    if (threadIdx.x < tileWidth && threadIdx.y < tileWidth) {
        for (int rowIndex = 0; rowIndex < maskWidth; rowIndex++) {
            for (int colIndex = 0; colIndex < maskWidth; colIndex++) {
                outputValue += sharedTile[threadIdx.y + rowIndex][threadIdx.x + colIndex] * maskConst[rowIndex][colIndex];
            } // end of inner for
        } // end of outer for
        // Set output value
        if (outputRow < imgHeight && outputCol < imgWidth) {
           //float clamped = min(max(outputValue, 0.0), 1.0);
            outputImage[(outputRow * imgWidth + outputCol) * imgChannels + curChannel] = outputValue;
        } // end of inner if
    } // end of outer if
} // end of kernel

int main(int argc, char *argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
//    float *deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputImageData,
                imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData,
                imageWidth * imageHeight * imageChannels * sizeof(float));
//    cudaMalloc((void **)&deviceMaskData,
//                maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
                imageWidth * imageHeight * imageChannels * sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maskConst, hostMaskData,
                maskWidth * maskWidth * sizeof(float));
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    
    //@@ INSERT CODE HERE
    wbLog(TRACE, "The input figure dimensions are", imageHeight, "x", imageWidth, "x", imageChannels);
  
    //@@ Initializing the grid and block dimensions
    dim3 DimBlock(blockWidth, blockWidth, 1);
    dim3 DimGrid((imageWidth - 1)/tileWidth + 1,(imageHeight - 1)/tileWidth + 1 , 1);
    
    //@@ Launching the GPU Kernel
    for (int curChannel = 0; curChannel < imageChannels; curChannel++) {
        wbLog(TRACE, "The current channel is ", curChannel);
        convKernelShared<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, curChannel, imageHeight, imageWidth, imageChannels);
    }
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                imageWidth * imageHeight * imageChannels * sizeof(float),
                cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
//    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
    }

