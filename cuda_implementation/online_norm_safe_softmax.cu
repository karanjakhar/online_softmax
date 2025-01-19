#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

__global__
void safeSoftmaxOnlineNorm(float * inputMatix,int rows, int cols, float * result){
    
    __shared__ float smem[1024];

    for(int row = blockIdx.x; row < rows; row += gridDim.x){
    int tid = threadIdx.x;

    if(row >= rows) return;

    float * input_row = inputMatix + row * cols;
    float * output_row = result + row * cols; 

    float local_max = -INFINITY;
    float curr_max = 0.0f;
    float local_norm = 0.0f;

    for(int j = tid; j < cols; j += blockDim.x){
        curr_max = fmax(input_row[j], local_max);
        local_norm = local_norm * expf(local_max - curr_max) + expf(input_row[j] - curr_max);
        local_max = curr_max;
        }



    // sync all threads and writing their local max to shared memory
    __syncthreads();

    smem[tid] = local_max;
    // sync again, so all threads complete writing 
    __syncthreads();
    

    // Now we start our reduction step

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride){
        smem[tid] = fmax(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smem[0];
    
    __syncthreads();


    // Now we will calculate row norm using reduction

    smem[tid] = local_norm * expf(local_max - row_max);

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride){

        smem[tid] += smem[tid + stride];
        }
        __syncthreads(); 
    }

    float row_norm = smem[0];

    __syncthreads();

    for(int i = tid; i < cols; i += blockDim.x){
        output_row[i] = expf(input_row[i] - row_max)/ row_norm;
    }
    }

}

void print_array(float * arr, int num_elements){
    

    for (int i = 0; i < num_elements; i++){
        printf("%f\t",arr[i]);
    }
    printf("\n");
}

void check_softmax_prob_sum(float * arr, int rows, int cols){
    float sum = 0.f;
    bool checkFailed = false;
    for(int i = 0; i < rows; i++){
        sum = 0.f;
        for(int j = 0; j < cols; j++){
            sum += arr[i*cols+j];
            // printf("sum: %f\n", sum);

        }
        if(sum > 1.2 || sum < 0.9){

            checkFailed = true;
            printf("Softmax check failed : sum of %d is %f \n", i+1, sum);
            return;
        }
        
        // printf("Sum for %d row:%f\n",i, sum );
    }
    if(!checkFailed) printf("Softmax check passed!!\n");
}


/*

The function uses the Box-Muller transform to convert two 
uniformly distributed random numbers (u1 and u2) into a 
normally distributed number (num) with mean 0 and std 1.


*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}


int main(){
    
    


    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
   
    int rows = 1024, cols = 32768;
    int matrixSize = rows * cols;
    size_t totalBytes = matrixSize * sizeof(float);
    float * inputMatrix = (float*)malloc(totalBytes);
    float * resultMatrix = (float*)malloc(totalBytes);

    if (inputMatrix == NULL){
        printf("Memory allocation failed\n");
        return;
    }


    printf("Num elements:%d\n", matrixSize);
    
    for(int i = 0; i < matrixSize; i++){
        inputMatrix[i] = random_normal_clamped(-10, 10);
    }
    
    

    float * inputMatrix_d, * resultMatrix_d;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&inputMatrix_d, totalBytes));
    CUDA_CHECK(cudaMalloc(&resultMatrix_d, totalBytes));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    cudaMemcpy(inputMatrix_d, inputMatrix, totalBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = 1024;
    printf("threadsPerBlock: %d\n", threadsPerBlock);
    safeSoftmaxOnlineNorm<<<blocksPerGrid,threadsPerBlock>>>(inputMatrix_d,rows, cols, resultMatrix_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);
    
    
    cudaEventRecord(start);
    cudaMemcpy(resultMatrix, resultMatrix_d, totalBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Device to host transfer time: %f ms\n", ms);





    check_softmax_prob_sum(resultMatrix, rows, cols);

    free(inputMatrix);
    free(resultMatrix);
    cudaFree(inputMatrix_d);
    cudaFree(resultMatrix_d);

    return 0;
}

