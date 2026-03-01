# Makefile
# CSE 554 Group 14
# Compiles the prof_example binary.

NVCC      := nvcc
TARGET    := prof_example
SRC       := prof_example.cu

NVCCFLAGS := -O2 -std=c++17
LDFLAGS   := -lcublas

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
