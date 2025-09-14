# Kompilatory
CC = gcc
NVCC = nvcc

# Flagi dla kompilatora C
CFLAGS = -Wall -Wextra -g -O2 -std=c99 -fopenmp

# Flagi dla kompilatora CUDA
NVFLAGS = -O2 -arch=native

# Flagi dla linkera. Używamy nvcc jako linkera, bo wie, jak dołączyć biblioteki CUDA.
# POPRAWKA: Zmieniamy -Xlinker na -Xcompiler dla flagi -fopenmp
LDFLAGS = -Xcompiler -fopenmp -lopenblas -lOpenCL -lrt -lcublas

# Nazwa pliku wykonywalnego
TARGET = program

# Automatycznie znajdź pliki źródłowe
SRCS_C = $(filter-out cuda_multiply.c, $(wildcard *.c))
SRCS_CU = $(wildcard *.cu)

# Zamień rozszerzenia na .o
OBJS_C = $(SRCS_C:.c=.o)
OBJS_CU = $(SRCS_CU:.cu=.o)
OBJS = $(OBJS_C) $(OBJS_CU)

# Cel domyślny
all: $(TARGET)

# Reguła linkowania: użyj nvcc do połączenia wszystkich plików .o
$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Reguła kompilacji dla plików .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Reguła kompilacji dla plików .cu
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Cel do czyszczenia
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean