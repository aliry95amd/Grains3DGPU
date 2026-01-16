#!/bin/bash

# Comprehensive collision detection benchmark script
# Runs all particle configurations with different shapes and sizes

# Exit on error
set -e

# Configuration
EXECUTABLE="./build/GJKPerformanceTest"
CSV_FILE="data/collision_benchmark_comprehensive.csv"
TRIALS=3
SEED=42

# Constants
r=0.05
A=$(echo "scale=10; c(1/3 * a(-1/64))" | bc -l)  # cos(1/3 * acos(-1/64)) ≈ 0.8634

# Box size for aspect ratio 1
BOX_SIZE=$(echo "scale=10; 2 / sqrt(3) * $r" | bc -l)

# Box size components for aspect ratio 4
BOX4_X=$(echo "scale=10; 1 / sqrt(6 * $A) * $r" | bc -l)
BOX4_Y=$(echo "scale=10; 1 / sqrt(6 * $A) * $r" | bc -l)
BOX4_Z=$(echo "scale=10; 16 / sqrt(3) / $A * $r" | bc -l)

# Superquadric size for aspect ratio 4
SQ4_X=$(echo "scale=10; $r / 2" | bc -l)
SQ4_Y=$(echo "scale=10; $r / 2" | bc -l)
SQ4_Z=$(echo "scale=10; 4 * $r" | bc -l)

# Create data directory if it doesn't exist
mkdir -p data

# Remove old CSV file to start fresh
rm -f "$CSV_FILE"

echo "================================================================================"
echo "Running Comprehensive Collision Detection Benchmark"
echo "================================================================================"
echo "Output file: $CSV_FILE"
echo "Trials per configuration: $TRIALS"
echo "Random seed: $SEED"
echo ""

# Counter for progress
total_configs=16
current=0

# Function to run benchmark
run_benchmark() {
    local particles=$1
    local domain=$2
    local shape=$3
    local size_x=$4
    local size_y=$5
    local size_z=$6
    local aspect=$7
    local append_flag=$8
    
    current=$((current + 1))
    echo "================================================================================"
    echo "Configuration $current/$total_configs"
    echo "  Particles: $particles"
    echo "  Domain: [$domain, $domain, $domain]"
    echo "  Shape: $shape"
    echo "  Size: [$size_x, $size_y, $size_z]"
    echo "  Aspect: $aspect"
    echo "================================================================================"
    
    # compute-sanitizer --tool memcheck 
    # cuda-gdb --args 
    $EXECUTABLE \
        --particles "$particles" \
        --domain "$domain" \
        --shape "$shape" \
        --size "$size_x" "$size_y" "$size_z" \
        --aspect "$aspect" \
        --precision both \
        --platform both \
        --trials "$TRIALS" \
        --seed "$SEED" \
        --csv "$CSV_FILE" \
        $append_flag
    
    echo ""
}

# =================================================================================
# Spheres - S1 (aspect ratio 1.0)
# =================================================================================
echo "Running Sphere benchmarks (S1)..."

# run_benchmark 512 $(echo "32 * $r" | bc -l) sphere $r $r $r 1.0 ""
# run_benchmark 2048 $(echo "48 * $r" | bc -l) sphere $r $r $r 1.0 "--append"
# run_benchmark 4096 $(echo "64 * $r" | bc -l) sphere $r $r $r 1.0 "--append"
# run_benchmark 8192 $(echo "80 * $r" | bc -l) sphere $r $r $r 1.0 "--append"
# run_benchmark 16384 $(echo "100 * $r" | bc -l) sphere $r $r $r 1.0 "--append"
# run_benchmark 32768 $(echo "112 * $r" | bc -l) sphere $r $r $r 1.0 "--append"
# run_benchmark 65584 $(echo "144 * $r" | bc -l) sphere $r $r $r 1.0 "--append"

# =================================================================================
# Boxes - B1 (aspect ratio 1.0)
# =================================================================================
echo "Running Box benchmarks (B1)..."

# run_benchmark 512 $(echo "32 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"
# run_benchmark 2048 $(echo "48 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"
# run_benchmark 4096 $(echo "64 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"
# run_benchmark 8192 $(echo "80 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"
# run_benchmark 16384 $(echo "100 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"
# run_benchmark 32768 $(echo "112 * $r" | bc -l) box $BOX_SIZE $BOX_SIZE $BOX_SIZE 1.0 "--append"

# =================================================================================
# Superquadrics - S4 (aspect ratio 4.0)
# =================================================================================
echo "Running Superquadric benchmarks (S4)..."

run_benchmark 512 $(echo "32 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"
run_benchmark 2048 $(echo "48 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"
run_benchmark 4096 $(echo "64 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"
run_benchmark 8192 $(echo "80 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"
run_benchmark 16384 $(echo "100 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"
run_benchmark 32768 $(echo "112 * $r" | bc -l) superquadric $SQ4_X $SQ4_Y $SQ4_Z 4.0 "--append"

# =================================================================================
# Boxes - B4 (aspect ratio 4.0)
# =================================================================================
echo "Running Box benchmarks (B4)..."

# run_benchmark 512 $(echo "32 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"
# run_benchmark 2048 $(echo "48 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"
# run_benchmark 4096 $(echo "64 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"
# run_benchmark 8192 $(echo "80 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"
# run_benchmark 16384 $(echo "100 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"
# run_benchmark 32768 $(echo "112 * $r" | bc -l) box $BOX4_X $BOX4_Y $BOX4_Z 4.0 "--append"

echo "================================================================================"
echo "All benchmarks completed!"
echo "Results saved to: $CSV_FILE"
echo "================================================================================"
