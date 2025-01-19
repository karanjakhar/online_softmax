#include <stdio.h>
#include <math.h>

void safeSoftmaxOnlineNorm(float * inputMatix,int num_elements, float * result){
    float m = -INFINITY;
    float new_m;
    float sum = 0;
    for(int i = 0; i < num_elements; i++){
        new_m = fmax(inputMatix[i], m);
        sum = sum * exp(m - new_m) + exp(inputMatix[i] - new_m);
        m = new_m;
    }

    
    
    // float sum = 0;
  
    // for(int i = 0; i < num_elements; i++ ){
    //     sum += exp(inputMatix[i] - max);
    // }

    for(int i = 0; i < num_elements; i++){
        result[i] = exp(inputMatix[i] - m)/sum;
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

    safeSoftmaxOnlineNorm(inputMatrix, num_elements, result);
    print_array(result, num_elements);

}