#removed below dependency:
#/usr/lib64/libboost_timer.so
#took out OpenMP for now, add back later -fopenmp

CPP=g++
CFLAGS=-I. -Wall -std=c++2A 
DEPS = ga.h


GyroAverage-OpenCL: GyroAverage.cpp ga.h  gautils.h
	$(CPP) GyroAverage.cpp   -std=c++2a -Wall   -O3  -I. -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL -I/usr/include/boost169 -lm -lfftw3

GyroAverage-CPU: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp   -std=c++2a -Wall  -O3  -I. -fopenmp -march=native -o GyroAverage-CPU  -I/usr/include/boost169 -lm -lfftw3

GyroAverage-CUDA:  GyroAverage.cpp ga.h gautils.h
	cp -f GyroAverage.cpp GyroAverage.cu 
	nvcc GyroAverage.cu -I. -O3  -o GyroAverage-CUDA -DVIENNACL_WITH_CUDA -lOpenCL -lfftw3 -I/usr/include/boost169   -Xcompiler -fopenmp -Xcompiler -I/usr/include/boost169 -DINCL_MATH_CONSTANTS=1  

GyroAverage-Home: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall -g -pg  -O3  -I. -fopenmp -march=native -o GyroAverage-Home   -lm -lfftw3

Home-Debug: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb  -I.  -o GyroAverage-Home   -lm -lfftw3

Home-Asan: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb -pg   -I. -fsanitize=address -fno-omit-frame-pointer -o GyroAverage-Home   -lm -lfftw3 -lasan


all: GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA

#lint: GyroAverage.cpp ga.h gautils.h
#	clang-tidy --checks=*,-readability-magic-numbers,-readability-isolate-declaration,-cppcoreguidelines-avoid-magic-numbers,-modernize-use-trailing-return-type  GyroAverage.cpp --extra-arg=-I. --header-filter=gautils.h

clean:
	rm -rf GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA GyroAverage-Home GyroAverage.cu ga.cu
