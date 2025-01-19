#include <stdio.h>
#include <math.h>

void naiveSoftmax(float * inputMatix,int num_elements, float * result){
    float sum = 0;
  
    for(int i = 0; i < num_elements; i++ ){
        sum += exp(inputMatix[i]);
    }

    for(int i = 0; i < num_elements; i++){
        result[i] = exp(inputMatix[i])/sum;
    }
}

void print_array(float * arr, int num_elements){
    

    for (int i = 0; i < num_elements; i++){
        printf("%f\t",arr[i]);
    }
    printf("\n");
}

void main(){
    float inputMatrix[5] = {400000.5, 800000.9, 10.0, 11.1, 12.4};
    int num_elements = sizeof(inputMatrix)/ sizeof(inputMatrix[0]);
    print_array(inputMatrix, num_elements);
    float result[5];

    naiveSoftmax(inputMatrix, num_elements, result);
    print_array(result, num_elements);

}