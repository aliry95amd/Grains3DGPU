
# Directories
BUILD_DIR = build
DATA_DIR = data

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
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/CDModuleBenchmark

# Default target
all: $(TARGET)

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

# Build target
$(TARGET): $(BUILD_DIR) $(DATA_DIR) $(OBJECTS)
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
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
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
	@rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned!"

clean-data:
	@rm -rf $(DATA_DIR)
	@echo "Data directory cleaned!"

clean-all: clean clean-data
	@echo "All directories cleaned!"

.PHONY: all test clean clean-data clean-all
