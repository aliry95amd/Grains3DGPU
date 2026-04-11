# Parent Makefile for GrainsGPU Validations
# This makefile orchestrates building all validation tools and tests

# Default target
.PHONY: all clean help collision-detection gjk-iterations contact-forces convergence-analysis visualization

# Build all validation tools
all: collision-detection gjk-iterations
	@echo "All validation tools built successfully!"

# Build collision detection validation tools
collision-detection:
	@echo "Building collision detection validation tools..."
	@if $(MAKE) -C CollisionDetection all; then \
		echo "Collision detection validation tools built successfully!"; \
	else \
		echo "Failed to build full validation tools. Building simple demo instead..."; \
		$(MAKE) -C CollisionDetection demo; \
		echo "Simple collision detection demo built successfully!"; \
	fi

# Build GJK iterations vs distance validation tools
gjk-iterations:
	@echo "Building GJK iterations validation tools..."
	@$(MAKE) -C GJKIterations all
	@echo "GJK iterations validation tools built successfully!"

# Build contact forces validation tools (placeholder for future implementation)
contact-forces:
	@echo "Contact forces validation tools not yet implemented"
	# @$(MAKE) -C ContactForces all

# Build convergence analysis validation tools (placeholder for future implementation)
convergence-analysis:
	@echo "Convergence analysis validation tools not yet implemented"
	# @$(MAKE) -C ConvergenceAnalysis all

# Build visualization tools (placeholder for future implementation)
visualization:
	@echo "Visualization validation tools not yet implemented"
	# @$(MAKE) -C Visualization all

# Clean all validation tools
clean:
	@echo "Cleaning collision detection validation tools..."
	@$(MAKE) -C CollisionDetection clean
	@echo "Cleaning GJK iterations validation tools..."
	@$(MAKE) -C GJKIterations clean
	@echo "All validation tools cleaned!"

# Run collision detection tests
test-collision:
	@echo "Running collision detection validation tests..."
	@if [ -f CollisionDetection/gjk_performance_test ]; then \
		$(MAKE) -C CollisionDetection test; \
	elif [ -f CollisionDetection/simple_gjk_demo ]; then \
		echo "Running simple demo instead..."; \
		$(MAKE) -C CollisionDetection test-demo; \
	else \
		echo "No test executable found. Please build first."; \
		exit 1; \
	fi

# Run GJK iterations vs distance test
test-gjk-iterations:
	@echo "Running GJK iterations validation test..."
	@$(MAKE) -C GJKIterations test

# Run all available tests
test: test-collision test-gjk-iterations
	@echo "All validation tests completed!"

# Install validation tools (copy executables to a common bin directory)
install:
	@echo "Installing validation tools..."
	@mkdir -p bin
	@if [ -f CollisionDetection/gjk_performance_test ]; then \
		cp CollisionDetection/gjk_performance_test bin/; \
		echo "Installed gjk_performance_test"; \
	fi
	@echo "Validation tools installed in bin/"

# Help target
help:
	@echo "GrainsGPU Validations Makefile"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all                      - Build all validation tools"
	@echo "  collision-detection      - Build collision detection validation tools"
	@echo "  collision-detection-demo - Build simple collision detection demo (no dependencies)"
	@echo "  contact-forces           - Build contact forces validation tools (future)"
	@echo "  convergence-analysis     - Build convergence analysis tools (future)"
	@echo "  visualization            - Build visualization tools (future)"
	@echo "  clean                   - Clean all validation tools"
	@echo "  test                    - Run all validation tests"
	@echo "  test-collision          - Run collision detection tests only"
	@echo "  install                 - Install validation executables to bin/"
	@echo "  help                    - Show this help message"
	@echo ""
	@echo "Individual validation directories:"
	@echo "  CollisionDetection/ - GJK performance comparison tools"
	@echo "  GJKIterations/      - GJK iteration count vs distance study"
	@echo "  ContactForces/      - Contact force model validation (future)"
	@echo "  ConvergenceAnalysis/ - Numerical convergence studies (future)"
	@echo "  Visualization/      - Plotting and analysis tools (future)"
