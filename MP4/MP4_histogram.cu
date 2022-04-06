// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
// Convert Float to Unsigned Char
__global__ void floatToUnsignedChar(float* input, unsigned char* output, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = blockIdx.z * (width * height) + y * width + x;
    output[index] = (unsigned char) ((HISTOGRAM_LENGTH - 1) * input[index]);
  }
}

// Convert Unsigned Char to Float
__global__ void unsignedCharToFloat(unsigned char* input, float* output, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = blockIdx.z * (width * height) + y * width + x;
    output[index] = (float) (input[index]/255.0);
  }
}

// Convert Image to Gray Scale
__global__ void imageToGrayScale(unsigned char* input, unsigned char* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int index = y * (width) + x;
    unsigned char r = input[3 * index + 0];
    unsigned char g = input[3 * index + 1];
    unsigned char b = input[3 * index + 2];
    output[index] = (unsigned char) ((0.21 * r + 0.71 * g + 0.07 * b));
  }
}

// Compute Histogram
__global__ void histogramKernel(unsigned char* input, unsigned int* output, int width, int height) {
  
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  
  int thread_index = threadIdx.x + blockDim.x * threadIdx.y;
  if (thread_index < HISTOGRAM_LENGTH) {
    histo_private[thread_index] = 0;
  }
  
  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int index = y * (width) + x;
    unsigned char position = input[index];
    atomicAdd(&(histo_private[position]), 1);
  }
  __syncthreads();
  
  if (thread_index < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[thread_index]), histo_private[thread_index]);
  }
}


// Compute the prefix sum of the histogram
__global__ void histogramToCumulativeDistr(unsigned int* input, float* output, int width, int height) {
  
  __shared__ unsigned int cumdistr[HISTOGRAM_LENGTH];
  
  cumdistr[threadIdx.x] = input[threadIdx.x];

  // First scan half
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < HISTOGRAM_LENGTH) {
      cumdistr[index] += cumdistr[index - stride];
    }
  }
  
  // Second scan half
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index + stride < HISTOGRAM_LENGTH) {
      cumdistr[index + stride] += cumdistr[index];
    }
  }
  
  __syncthreads();

  output[threadIdx.x] = cumdistr[threadIdx.x] / ((float) (width * height));
}

// Histogram Equalization Function
__global__ void equalizeHistogram(unsigned char* input, float* cumulativeDistr, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = blockIdx.z * (width * height) + y * width + x;
    
    float equalized = 255.0*(cumulativeDistr[input[index]] - cumulativeDistr[0])/(1.0 - cumulativeDistr[0]);
    float clamped = min(max(equalized, 0.0), 255.0);

    input[index] = (unsigned char) (clamped);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float   *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceGrayImageChar;
  unsigned int  *deviceGrayImageHist;
  float   *deviceImageCumulativeDistr;
      
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  // Allocating GPU memory
  wbTime_start(Generic, "Allocating memory and copying data to GPU");
  cudaMalloc((void **) &deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceImageChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImageChar, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImageHist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  // cudaMemset((void *) deviceGrayImageHist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceImageCumulativeDistr, HISTOGRAM_LENGTH * sizeof(float));
      
  // Copy data to GPU
  cudaMemcpy(deviceImageFloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Generic, "Allocating memory and copying data to GPU");
  
  dim3 dimGrid;
  dim3 dimBlock;
  
  // Convert from float to unsigned char
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32,32,1);
  floatToUnsignedChar<<<dimGrid,dimBlock>>>(deviceImageFloat, deviceImageChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();
      
  // Convert from RGB to Gray scale
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32,32,1);
  imageToGrayScale<<<dimGrid,dimBlock>>>(deviceImageChar, deviceGrayImageChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();
      
  // Compute the histogram of gray image
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32,32,1);
  histogramKernel<<<dimGrid,dimBlock>>>(deviceGrayImageChar, deviceGrayImageHist, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  // Compute the min value of the cumulative distribution function
  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH,1,1);
  histogramToCumulativeDistr<<<dimGrid,dimBlock>>>(deviceGrayImageHist, deviceImageCumulativeDistr, imageWidth, imageHeight);
  cudaDeviceSynchronize();
         
  // Compute the histogram equalization function 
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32,32,1);
  equalizeHistogram<<<dimGrid,dimBlock>>>(deviceImageChar, deviceImageCumulativeDistr, imageWidth, imageHeight);
  cudaDeviceSynchronize(); 
      
  // Cast equalized histogram back to float 
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32,32,1);
  unsignedCharToFloat<<<dimGrid,dimBlock>>>(deviceImageChar, deviceImageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize(); 
      
  // Copy data to Host
  cudaMemcpy(hostOutputImageData, deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
   
  
  wbTime_start(Generic, "Releasing GPU memory");  
  // Free GPU Memory
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceGrayImageChar);
  cudaFree(deviceGrayImageHist);
  cudaFree(deviceImageCumulativeDistr);
  wbTime_stop(Generic, "Releasing GPU memory");  
      
  // Check solution
  // wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here

  wbTime_start(Generic, "Releasing Host memory");  
  // Free GPU Memory
  free(hostInputImageData);
  free(hostOutputImageData);
  wbTime_stop(Generic, "Releasing Host memory");  
  return 0;
}

