#include <stdio.h>
#include <math.h>

void safeSoftmax(float * inputMatix,int num_elements, float * result){
    float max = -INFINITY;

    for(int i = 0; i < num_elements; i++){
        if(inputMatix[i] > max){
            max = inputMatix[i];
        }
    }
    
    
    float sum = 0;
  
    for(int i = 0; i < num_elements; i++ ){
        sum += exp(inputMatix[i] - max);
    }

    for(int i = 0; i < num_elements; i++){
        result[i] = exp(inputMatix[i] - max)/sum;
    }
}

void print_array(float * arr, int num_elements){
    

    for (int i = 0; i < num_elements; i++){
        printf("%f\t",arr[i]);
    }
    printf("\n");
}

void main(){
    float inputMatrix[5] = {940005.5, 940004.9, 940003.0, 940002.1, 940001.4};
    int num_elements = sizeof(inputMatrix)/ sizeof(inputMatrix[0]);
    print_array(inputMatrix, num_elements);
    float result[5];

    safeSoftmax(inputMatrix, num_elements, result);
    print_array(result, num_elements);

}