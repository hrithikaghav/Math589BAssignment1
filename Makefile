# Define compilers
CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -fPIC
CXXFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. -lm

# Output libraries and executables
LIBRARY = energy_bfgs.so
OBJECTS = energy_bfgs.o

# Rule to build the shared library
$(LIBRARY): $(OBJECTS)
	$(CC) -shared -o $(LIBRARY) $(OBJECTS) -fPIC

# Compile the C code into an object file
energy_bfgs.o: energy_bfgs.c
	$(CC) $(CFLAGS) -c energy_bfgs.c -o energy_bfgs.o

# C++ shared library for energy calculations
libenergy.so: energy.cpp energy.hpp
	$(CXX) $(CXXFLAGS) -shared -o libenergy.so -fPIC energy.cpp

# Gradients with Armijo method (C++)
grad_w_armijo: libenergy.so grad_w_armijo.o
	$(CXX) $(CXXFLAGS) grad_w_armijo.o -o grad_w_armijo $(LDFLAGS) -lenergy

# BFGS with classes (C++)
bfgs_w_classes: bfgs_w_classes.o
	$(CXX) $(CXXFLAGS) bfgs_w_classes.o -o bfgs_w_classes $(LDFLAGS)

# BFGS with variable arguments (C)
bfgs_w_varargs: bfgs_w_varargs.o
	$(CC) $(CFLAGS) bfgs_w_varargs.o -o bfgs_w_varargs $(LDFLAGS)

# Clean rule to remove generated files
clean:
	@rm -f $(LIBRARY) grad_w_armijo bfgs_w_classes bfgs_w_varargs *.o

# Force target to ensure clean works without errors
FORCE:
