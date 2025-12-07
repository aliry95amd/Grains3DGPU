#!/bin/bash

# Google Test Installation Script for GrainsGPU

set -e

echo "=== Installing Google Test Dependencies ==="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Update package manager
echo "Updating package manager..."
$SUDO apt-get update

# Install basic dependencies
echo "Installing basic dependencies..."
$SUDO apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtest-dev \
    libgmock-dev \
    pkg-config

# Check if GTest is properly installed
GTEST_SOURCE_DIR="/usr/src/googletest"
GTEST_BUILD_DIR="/tmp/googletest-build"

if [ ! -d "$GTEST_SOURCE_DIR" ]; then
    echo "Google Test source not found. Installing from repository..."
    
    # Clone and build Google Test manually
    cd /tmp
    git clone https://github.com/google/googletest.git
    cd googletest
    
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DINSTALL_GTEST=ON
    
    make -j$(nproc)
    $SUDO make install
    
    # Update library cache
    $SUDO ldconfig
    
    echo "Google Test installed successfully"
else
    echo "Building Google Test from system sources..."
    
    # Build from system sources
    mkdir -p "$GTEST_BUILD_DIR"
    cd "$GTEST_BUILD_DIR"
    
    cmake "$GTEST_SOURCE_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DINSTALL_GTEST=ON
    
    make -j$(nproc)
    $SUDO make install
    
    # Update library cache
    $SUDO ldconfig
    
    echo "Google Test built and installed successfully"
fi

# Verify installation
echo "Verifying Google Test installation..."

# Check for headers
if [ -f "/usr/local/include/gtest/gtest.h" ] || [ -f "/usr/include/gtest/gtest.h" ]; then
    echo "✓ Google Test headers found"
else
    echo "✗ Google Test headers not found"
    exit 1
fi

# Check for libraries
if ldconfig -p | grep -q "libgtest" && ldconfig -p | grep -q "libgtest_main"; then
    echo "✓ Google Test libraries found"
else
    echo "✗ Google Test libraries not found"
    echo "Available gtest libraries:"
    ldconfig -p | grep gtest || echo "None found"
    exit 1
fi

# Install CUDA if requested
if [[ "$1" == "--with-cuda" ]]; then
    echo "Installing CUDA dependencies..."
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        echo "CUDA already installed: $(nvcc --version | grep release)"
    else
        echo "Installing CUDA toolkit..."
        
        # Add NVIDIA package repository
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-keyring_1.0-1_all.deb
        $SUDO dpkg -i cuda-keyring_1.0-1_all.deb
        $SUDO apt-get update
        
        # Install CUDA toolkit
        $SUDO apt-get install -y cuda-toolkit-12-0
        
        # Add CUDA to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        
        echo "CUDA installed. Please restart your terminal or run:"
        echo "source ~/.bashrc"
    fi
fi

echo "=== Installation Complete ==="
echo ""
echo "You can now build and run tests with:"
echo "cd Tests"
echo "mkdir build && cd build"
echo "cmake .."
echo "make"
echo "./grains_tests"
