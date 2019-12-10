#include <iostream>
#include <omp.h>

int main(){
  #pragma omp parallel
  std::cout << "Test";
  
}
