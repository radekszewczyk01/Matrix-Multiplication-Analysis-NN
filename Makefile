# Nazwa kompilatora C
CC = gcc

# Flagi kompilatora:
# Dodajemy -fopenmp tutaj, aby kompilator rozumiał pragmy OpenMP
CFLAGS = -Wall -Wextra -g -O2 -std=c99 -fopenmp

# Flagi linkera:
# -lopenblas, -lOpenCL, -lrt to biblioteki do dołączenia na końcu
LDFLAGS = -lopenblas -lOpenCL -lrt -fopenmp

# Nazwa pliku wykonywalnego
TARGET = program

# Automatycznie znajdź wszystkie pliki źródłowe .c w bieżącym katalogu
SRCS = $(wildcard *.c)
# Zamień rozszerzenia .c na .o
OBJS = $(SRCS:.c=.o)

# Cel domyślny
all: $(TARGET)
	@echo "Build complete. Cleaning up object files"
	$(MAKE) clean_objs
# Reguła linkowania
# Dodajemy flagi linkera na końcu polecenia
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Reguła kompilacji
# Ta reguła używa CFLAGS do kompilacji każdego pliku .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean_objs:
	rm -f $(OBJS)

# Cel do czyszczenia
clean:
	rm -f $(TARGET) $(OBJS)

# Informuje make, że "all" i "clean" to nie są nazwy plików
.PHONY: all clean