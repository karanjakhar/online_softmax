#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>



void safeSoftmaxOnlineNorm(float * inputMatix,int rows, int cols, float * result){
    
    
    
    for(int i = 0; i < rows; i++){
        float m = -INFINITY;
        float new_m;
        float sum = 0.f;

        for(int j = 0; j < cols; j++){
        new_m = fmax(inputMatix[i*cols+j], m);
        sum = sum * exp(m - new_m) + exp(inputMatix[i*cols+j] - new_m);
        m = new_m;
    }

     for(int j = 0; j < cols; j++){
        result[i*cols+j] = exp(inputMatix[i*cols+j] - m)/sum;
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
        }
        if(sum > 1.2 || sum < 0.9){
            checkFailed = true;
            printf("Softmax check failed.\n");
            return;
        }
        
        // printf("Sum for %d row:%f\n",i, sum );
    }
    printf("Softmax check passed!!\n");
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



void main(){
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); 

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
        inputMatrix[i] = random_normal_clamped(-100, 100);
    }
    

    safeSoftmaxOnlineNorm(inputMatrix,rows, cols, resultMatrix);
    check_softmax_prob_sum(resultMatrix, rows, cols);

    free(inputMatrix);
    free(resultMatrix);

    clock_gettime(CLOCK_MONOTONIC, &end);   // End time

    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9; // Seconds + nanoseconds
    printf("Execution Time: %f seconds\n", time_taken);


}