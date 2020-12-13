#removed below dependency:
#/usr/lib64/libboost_timer.so
#took out OpenMP for now, add back later -fopenmp

CPP = g++
CFLAGS=-I. -Wall -std=c++2A 
DEPS = ga.h
LIBS = -lm  -lboost_program_options -lboost_iostreams -lgsl
FFTWLIBS = -lfftw3 -lfftw3f

GyroAverage-OpenCL: GyroAverage.cpp ga.h  gautils.h
	$(CPP) GyroAverage.cpp   -std=c++14 -Wall   -O3  -I. -D_GLIBCXX_USE_CXX11_ABI=0  -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL -I/usr/include/boost169 $(LIBS) $(FFTWLIBS)

GyroAverage-CPU: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp   -std=c++14 -Wall  -D_GLIBCXX_USE_CXX11_ABI=0 -O3  -I. -fopenmp -march=native -o GyroAverage-CPU  -I/usr/include/boost169  $(FFTWLIBS)

GyroAverage-CUDA:  GyroAverage.cpp ga.h gautils.h
	cp -f GyroAverage.cpp GyroAverage.cu 
	nvcc GyroAverage.cu  -std=c++14 -I. -O3  -o GyroAverage-CUDA -D_GLIBCXX_USE_CXX11_ABI=0  -DVIENNACL_WITH_CUDA -lboost_program_options -lOpenCL $(LIBS) $(FFTWLIBS) -I/usr/include/boost169   -Xcompiler -fopenmp -Xcompiler -I/usr/include/boost169 -DINCL_MATH_CONSTANTS=1  



GyroAverage-Prince-CPU: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++14 -Wall  -O3   -fopenmp -march=native -o GyroAverage-CPU  -I${BOOST_INC} -I.    -L${BOOST_LIB}   $(LIBS) $(FFTWLIBS)


GyroAverage-Prince-OpenCL: GyroAverage.cpp ga.h  gautils.h
	$(CPP) GyroAverage.cpp -std=c++14  -Wall   -O3   -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL  -I${BOOST_INC}  -I.   -L${BOOST_LIB} $(LIBS) $(FFTWLIBS)
GyroAverage-Prince-CUDA:  GyroAverage.cpp ga.h gautils.h
	cp -f GyroAverage.cpp GyroAverage.cu
	nvcc GyroAverage.cu    -I. -O3  -o GyroAverage-CUDA   -DVIENNACL_WITH_CUDA  -L{BOOST_LIB}  -lboost_program_options -lOpenCL $(LIBS)   $(FFTWLIBS)  -Xcompiler -fopenmp   -I${CUDA_INC}  -I${BOOST_INC} -I${CUDA_INC}/crt     -L${BOOST_LIB} -DINCL_MATH_CONSTANTS=1

GyroAverage-Greene-CPU: GyroAverage.cpp ga.h gautils.h
	icc GyroAverage.cpp  -std=c++14 -Wall  -O3   -fopenmp -march=native -o GyroAverage-CPU  -I${BOOST_INC} -I.  -I/share/apps/intel/19.1.2/mkl/include/fftw   -L${BOOST_LIB}   $(LIBS) -mkl


GyroAverage-Greene-OpenCL: GyroAverage.cpp ga.h  gautils.h
	icc GyroAverage.cpp -std=c++14  -Wall   -O3   -fopenmp -march=native -o GyroAverage-OpenCL -DVIENNACL_WITH_OPENCL -lOpenCL  -I${BOOST_INC}  -I.  -I/share/apps/intel/19.1.2/mkl/include/fftw  -L${BOOST_LIB} $(LIBS) -mkl
GyroAverage-Greene-CUDA:  GyroAverage.cpp ga.h gautils.h
	cp -f GyroAverage.cpp GyroAverage.cu
	nvcc GyroAverage.cu    -I. -O3  -o GyroAverage-CUDA   -DVIENNACL_WITH_CUDA  -L{BOOST_LIB}  -ccbin=icc -lboost_program_options -lOpenCL $(LIBS)   -Xcompiler -fopenmp    -I/share/apps/intel/19.1.2/mkl/include/fftw -I${CUDA_INC}  -I${BOOST_INC} -I${CUDA_INC}/crt     -L${BOOST_LIB} 
-DINCL_MATH_CONSTANTS=1 -mkl

Greene: GyroAverage-Greene-CPU GyroAverage-Greene-OpenCL GyroAverage-Greene-CUDA





GyroAverage-Home: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall -g -pg  -O3  -I. -fopenmp -march=native -o GyroAverage-Home   $(LIBS)

Home-Debug: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb  -I.  -o GyroAverage-Home  $(LIBS)

Home-Asan: GyroAverage.cpp ga.h gautils.h
	$(CPP) GyroAverage.cpp  -std=c++2a -Wall   -ggdb -O1   -fopenmp -I. -fsanitize=address -fno-omit-frame-pointer -o GyroAverage-Home  $(LIBS)

GyroAverage-Home-Clang: GyroAverage.cpp ga.h gautils.h
	clang++ GyroAverage.cpp  -std=c++2a -Wall -g -pg  -O3  -I. -fopenmp -march=native -o GyroAverage-Home-Clang   $(LIBS) -lstdc++ 


all: GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA

Prince:  GyroAverage-Prince-CPU GyroAverage-Prince-OpenCL GyroAverage-Prince-CUDA

#lint: GyroAverage.cpp ga.h gautils.h
#	clang-tidy --checks=*,-readability-magic-numbers,-readability-isolate-declaration,-cppcoreguidelines-avoid-magic-numbers,-modernize-use-trailing-return-type  GyroAverage.cpp --extra-arg=-I. --header-filter=gautils.h

clean:
	rm -rf GyroAverage-OpenCL GyroAverage-CPU GyroAverage-CUDA GyroAverage-Home GyroAverage.cu ga.cu
