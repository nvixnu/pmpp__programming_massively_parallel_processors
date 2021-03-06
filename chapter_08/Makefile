CC = nvcc
OBJDIR = ./obj

DEPSNAMES = \
nvixnu__array_utils \
nvixnu__populate_arrays_utils \
nvixnu__error_utils \
pmpp__prefix_sum \
nvixnu__cuda_devices_props

CFLAGS = -g -G --compiler-options -Wall -lopencv_imgcodecs -lopencv_core

INCLUDES = $(patsubst %,-I ../../%, $(DEPSNAMES))

# List with all .cu files inside $(REPOSDIR)/<repoName>
CUFILES = $(foreach dep,$(DEPSNAMES), $(wildcard ../../$(dep)/*.cu))

# List with all .o paths
OBJS = $(patsubst %.cu,%.o,$(CUFILES))

# Compiled objects path
COMPILEDOBJS := $(patsubst %,$(OBJDIR)/%,$(notdir $(OBJS))) 

# Creates the obj dir, compiles each dependency and then the main app
all: objdir $(OBJS)
	nvcc ch8__full_prefix_sum.cu -o full_prefix_sum.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)
	nvcc ch8__partial_prefix_sum.cu -o partial_prefix_sum.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

# Creates the ./obj dir
objdir:
	mkdir	-p	$(OBJDIR)

# Compile a dependency
%.o: %.cu
	nvcc -c $< -o $(OBJDIR)/$(notdir $@) $(CFLAGS)

# Run the executable
run:
	./main

# Remove the generated artifacts
clean:
	rm -Rf $(OBJDIR)/*.o
	rm -Rf 1d_convolution.out
	rm -Rf partial_prefix_sum.out

.PHONY: all clean app run objdir