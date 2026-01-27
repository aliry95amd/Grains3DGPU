
# Reuse CollisionDetection Makefile conventions for building benchmarks

BUILD_DIR = build
DATA_DIR = data

INCLUDE_DIRS = $(GRAINS_INCDIR) $(GRAINS_XERCES_INCDIR)
INCFLAGS := $(INCLUDE_DIRS:%=-I%)
GRAINS_OBJECTS = $(wildcard $(GRAINS_OBJDIR)/*.o)
LIB_DIRS = -L$(GRAINS_XERCES_LIBDIR)
LIBFLAGS = -lxerces-c -lcudart
COMPFLAGS := $(GRAINS_GPU_COMPILER_FLAGS) $(INCFLAGS)
LINKFLAGS := $(GRAINS_GPU_LINKER_FLAGS) $(GRAINS_XERCES_FLAGS)


SOURCES = main.cpp
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/ContactTablePerformance

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)


$(TARGET): $(BUILD_DIR) $(DATA_DIR) $(OBJECTS)
	@if [ ! -d "$(GRAINS_OBJDIR)" ]; then \
		echo "Error: Grains object files not found at $(GRAINS_OBJDIR)"; \
		exit 1; \
	fi
	@echo "Linking ContactTablePerformance against Grains object files..."
	@$(GRAINS_GPU_COMPILER) $(LINKFLAGS) $(OBJECTS) $(GRAINS_OBJECTS) $(LIB_DIRS) $(LIBFLAGS) -o $@
	@echo "$(TARGET) build successful!"


$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@echo "Compiling $<..."
	@$(GRAINS_GPU_COMPILER) $(COMPFLAGS) -MMD -MP -c $< -o $@

test: $(TARGET)
	@if [ -f "./$(TARGET)" ]; then \
		./$(TARGET); \
	else \
		echo "Error: $(TARGET) not found. Please build first using 'make all'"; \
		exit 1; \
	fi

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned!"

clean-data:
	@rm -rf $(DATA_DIR)
	@echo "Data directory cleaned!"

clean-all: clean clean-data
	@echo "All directories cleaned!"

.PHONY: all test clean clean-data clean-all
