NVCC      = nvcc
LIBS      = -lcudart
TARGET    = a.out
OBJS      = main.o bitonic.o

# --- Help Menu ---

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build the application with optimizations (default)"
	@echo "  debug     Build with debug symbols (-g -G) for cuda-gdb"
	@echo "  clean     Remove object files and executable"
	@echo "  help      Show this help message"

# --- Rules ---

all: $(TARGET)

# The 'debug' target
# This must be called BEFORE 'all' (e.g., 'make debug')
debug: NVCCFLAGS = -g -G
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean debug