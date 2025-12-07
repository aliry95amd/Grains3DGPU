
# Paths
INCLUDE_DIRS = $(GRAINS_INCDIR) $(GRAINS_XERCES_INCDIR)
INCFLAGS  := $(INCLUDE_DIRS:%=-I%)
GRAINS_OBJECTS = $(wildcard $(GRAINS_OBJDIR)/*.o)
LIB_DIRS = -L$(GRAINS_XERCES_LIBDIR)
LIBFLAGS = -lxerces-c -lcudart -lcurand
COMPFLAGS := $(GRAINS_GPU_COMPILER_FLAGS) $(INCFLAGS)
LINKFLAGS := $(GRAINS_GPU_LINKER_FLAGS) $(GRAINS_XERCES_FLAGS)

# Source files
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = GJKPerformanceTest

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	@if [ ! -d "$(GRAINS_OBJDIR)" ]; then \
		echo "Error: Grains object files not found at $(GRAINS_OBJDIR)"; \
		exit 1; \
	fi
	@if [ ! -d "$(GRAINS_XERCES_LIBDIR)" ]; then \
		echo "Error: Xerces library not found at $(GRAINS_XERCES_LIBDIR)"; \
		exit 1; \
	fi
	@echo "Linking against Grains object files..."
	@$(GRAINS_GPU_COMPILER) $(LINKFLAGS) $(OBJECTS) $(GRAINS_OBJECTS) $(LIB_DIRS) $(LIBFLAGS) -o $@
	@echo "$(TARGET) build successful!"

# Compile source files
%.o: %.cpp
	@echo "Compiling $<..."
	@$(GRAINS_GPU_COMPILER) $(COMPFLAGS) -MMD -MP -D_XML -c $< -o $@

# Run tests
test: $(TARGET)
	@if [ -f "./$(TARGET)" ]; then \
		./$(TARGET); \
	else \
		echo "Error: $(TARGET) not found. Please build first using 'make all'"; \
		exit 1; \
	fi

# Clean
clean:
	@rm -f $(OBJECTS) $(TARGET) *.d
	@echo "${TARGET} cleaned!"

.PHONY: all test clean
