CC = nvcc
OBJDIR = ./obj

DEPSNAMES = nvixnu__error_utils

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
	nvcc ch3__blur.cu -o blur.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)
	nvcc ch3__color_to_grayscale.cu -o color_to_grayscale.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

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
	rm -Rf blur.out
	rm -Rf color_to_grayscale.out

.PHONY: all clean app run