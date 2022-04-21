#Compiler flag for architecture
ifndef ARCH
ARCH = 64
endif

#COMPILER_FLAGS specifies the additional compilation options we're using
CFLAGS = -O3 -m64

#EXECUTABLE specifies the name 
EXE = MonteCarlo

#CC specifies which compiler we're using
CC = g++

#NVCC specifies alternative Nvidia compiler we're using
NVCC = nvcc 

#INCLUDE LIBRARIES specifies the libraries we're linking against
INC	 =

#LIBRARY_PATH specifies the additional compilation options:
LPATH =

#LIBRARY_PATH specifies the additional compilation options: -lblas -lgfortran
LIBS = -lcufft

#LIBRARY_PATH specifies the additional compilation options: -lblas -lgfortran
NVARCH = 
NVFLAGS = -Xptxas -O3

#OBJS specifies which files to compile as part of the project
TARGS	 = HostFunctions.o DeviceKernels.o ElectricField.o main.o 

#====================================================================================================
# Make Compilation Options:
#====================================================================================================
all   : $(EXE)

%.o   : %.cpp
	$(CC) $(CFLAGS) $(INC) -c  $? 

%.o   : %.cu
	$(NVCC) $(NVFLAGS) $(INC) -c $? 

$(EXE): $(TARGS)
	$(NVCC) $(NVFLAGS) $(LPATH) $(LIBS) $(TARGS) $(INC) -o $@

clean : 
	rm -rf $(EXE) *.o
