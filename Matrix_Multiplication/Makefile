# Compiler
CC := mpicc

# Compiler flags
CFLAGS := -O3 -lopenblas -lcudart -lcublas -fopenmp -march=native -Wall -Minfo=all

# Source files
SRCS := $(wildcard *.c)

# Object files
OBJDIR := obj
OBJS := $(SRCS:.c=.o)

# Target executable
TARGET := run.x

# Default target
all: $(TARGET)

# Link object files into the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Compile source files into object files
$(OBJDIR)/*.o: $(SRCDIR)/*.c
	$(CC) $(CFLAGS) $^ -o $@

# Clean up object files and the target executable
clean:
	rm -f $(OBJDIR)/*.o $(TARGET) ./*.o