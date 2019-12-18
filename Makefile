CPP=g++
CFLAGS=-I.
DEPS = ga.h


GyroAverage-OpenCL: GyroAverage.cpp ga.h 
	$(CPP) GyroAverage.cpp  -Wall -Wextra  -O3  -I. -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL -I/usr/include/boost169 /usr/lib64/libboost_timer.so

GyroAverage-CPU: GyroAverage.cpp ga.h
	$(CPP) GyroAverage.cpp  -Wall   -Wextra -O3  -I. -fopenmp -march=native -o GyroAverage-CPU  -lOpenCL -I/usr/include/boost169 /usr/lib64/libboost_timer.so

GyroAverage-CUDA: GyroAverage.cu GyroAverage.cpp ga.h
	nvcc GyroAverage.cu -I. -O3  -o GyroAverage-CUDA -DVIENNACL_WITH_CUDA -lOpenCL -I/usr/include/boost169  /usr/lib64/libboost_timer.so -Xcompiler -fopenmp

all: GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA

clean:
	rm -rf GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA
