#removed below dependency:
#/usr/lib64/libboost_timer.so
#took out OpenMP for now, add back later -fopenmp

CPP=g++
CFLAGS=-I. -Wall -std=c++2A 
DEPS = ga.h


GyroAverage-OpenCL: GyroAverage.cpp ga.h  gautils.h
	$(CPP) GyroAverage.cpp   -std=c++14 -Wall   -O3  -I. -D_GLIBCXX_USE_CXX11_ABI=0  -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL -I/usr/include/boost169 -lm -lfftw3 -lfftw3f -lboost_program_options

GyroAverage-CPU: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp   -std=c++14 -Wall  -D_GLIBCXX_USE_CXX11_ABI=0 -O3  -I. -fopenmp -march=native -o GyroAverage-CPU  -I/usr/include/boost169 -lm -lfftw3 -lfftw3f -lboost_program_options

GyroAverage-CUDA:  GyroAverage.cpp ga.h gautils.h
	cp -f GyroAverage.cpp GyroAverage.cu 
	nvcc GyroAverage.cu  -std=c++14 -I. -O3  -o GyroAverage-CUDA -D_GLIBCXX_USE_CXX11_ABI=0  -DVIENNACL_WITH_CUDA -lboost_program_options -lOpenCL -lfftw3 -lfftw3f -I/usr/include/boost169   -Xcompiler -fopenmp -Xcompiler -I/usr/include/boost169 -DINCL_MATH_CONSTANTS=1  


GyroAverage-Prince-CPU: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp   -Wall  -O3  -I. -fopenmp -march=native -o GyroAverage-CPU  -lm -lfftw3 -lfftw3f -I${BOOST_INC}  -I${EIGEN_INC}   -L${BOOST_LIB}   -lboost_program_options


GyroAverage-Prince-OpenCL: GyroAverage.cpp ga.h  gautils.h
	$(CPP) GyroAverage.cpp   -std=c++2a -Wall   -O3  -I. -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL  -I${BOOST_INC}  -I${EIGEN_INC}   -L${BOOST_LIB} -lm -lfftw3 -lfftw3f -lboost_program_options



GyroAverage-Home: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall -g -pg  -O3  -I. -fopenmp -march=native -o GyroAverage-Home   -lm -lfftw3 -lfftw3f -lboost_program_options

Home-Debug: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb  -I.  -o GyroAverage-Home   -lm -lfftw3 -lfftw3f -lboost_program_options

Home-Asan: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb -O1   -fopenmp -I. -fsanitize=address -fno-omit-frame-pointer -o GyroAverage-Home   -lm -lfftw3 -lfftw3f -lasan  -lboost_program_options

GyroAverage-Home-Clang: GyroAverage.cpp ga.h gautils.h
	clang++ GyroAverage.cpp  -std=c++2a -Wall -g -pg  -O3  -I. -fopenmp -march=native -o GyroAverage-Home-Clang   -lm -lfftw3 -lfftw3f -lstdc++ -lboost_program_options


all: GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA

#lint: GyroAverage.cpp ga.h gautils.h
#	clang-tidy --checks=*,-readability-magic-numbers,-readability-isolate-declaration,-cppcoreguidelines-avoid-magic-numbers,-modernize-use-trailing-return-type  GyroAverage.cpp --extra-arg=-I. --header-filter=gautils.h

clean:
	rm -rf GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA GyroAverage-Home GyroAverage.cu ga.cu
