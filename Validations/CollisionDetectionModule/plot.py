#!/usr/bin/env python3
"""
Visualize Collision Detection Benchmark Results
Complex multi-subplot visualization showing optimal algorithm configurations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure plot style
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 16})

# Configurable colors
CPU_COLOR = '#4A90E2'  # Light blue for CPU
GPU_COLOR = '#1E3A8A'  # Dark blue for GPU

# 8 shades of blue for 8 combinations
COMBO_COLORS = [
    '#E3F2FD',  # Lightest blue
    '#BBDEFB',
    '#90CAF9',
    '#64B5F6',
    '#42A5F5',
    '#2196F3',
    '#1E88E5',
    '#1565C0'   # Darkest blue
]

def create_shape_label(row):
    """Create shape label from ShapeType and AspectRatio"""
    shape = row['ShapeType']
    aspect = int(row['AspectRatio'])
    
    if shape == 'Sphere':
        return f'S{aspect}'
    elif shape == 'Box':
        return f'B{aspect}'
    elif shape == 'Superquadric':
        return f'SQ{aspect}'
    return shape

def create_combo_code(row):
    """Create abbreviation for algorithm combination"""
    # GJK Algorithm: SignedVolume (S) or Johnson (J)
    algo = 'S' if row['GJKAlgo'] == 'SignedVolume' else 'J'
    
    # Representation: Transform (T) or Quaternion (Q)
    rep = 'T' if row['GJKRepresentation'] == 'Transform' else 'Q'
    
    # Transform type: Relative (R) or Absolute (A)
    trans = 'R' if row['UseRelativeTransform'] == 1 else 'A'
    
    return f'{algo}{rep}{trans}'

def find_best_configuration(df, precision, platform, particle_count, shape_label):
    """Find the best algorithm configuration for given parameters"""
    # Filter data
    subset = df[
        (df['Precision'] == precision) & 
        (df['Platform'] == platform) & 
        (df['ParticleCount'] == particle_count) &
        (df['ShapeLabel'] == shape_label)
    ].copy()
    
    if len(subset) == 0:
        return None, None, None
    
    # Add combo code
    subset['ComboCode'] = subset.apply(create_combo_code, axis=1)
    
    # Compute median for each combination
    combo_medians = subset.groupby('ComboCode')['TotalTime_ms'].median()
    
    # Find best (minimum time) configuration
    best_combo = combo_medians.idxmin()
    best_time = combo_medians.min()
    
    # Get pair count (should be same for all rows with these params)
    pair_count = subset['PairCount'].iloc[0]
    
    return best_combo, best_time, pair_count

def plot_precision(csv_file, precision, output_file):
    """Create 2x2 subplot for a given precision (Single or Double)"""
    # Read data
    df = pd.read_csv(csv_file)
    
    # Add shape label
    df['ShapeLabel'] = df.apply(create_shape_label, axis=1)
    
    # Define shape categories
    shapes = ['S1', 'B1', 'SQ4', 'B4']
    shape_titles = {
        'S1': 'Sphere (AR=1)',
        'B1': 'Box (AR=1)',
        'SQ4': 'Superquadric (AR=4)',
        'B4': 'Box (AR=4)'
    }
    
    # Get unique particle counts (reversed: smallest on top, largest on bottom)
    particle_counts = sorted(df['ParticleCount'].unique(), reverse=True)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Collision Detection Performance - {precision} Precision', fontsize=18, y=0.995)
    
    # First pass: collect all data to determine global x-axis range
    all_times = []
    shape_data = {}
    
    for shape in shapes:
        cpu_times = []
        gpu_times = []
        cpu_labels = []
        gpu_labels = []
        y_labels = []
        
        for pc in particle_counts:
            # Find best CPU configuration
            cpu_combo, cpu_time, pair_count = find_best_configuration(
                df, precision, 'CPU', pc, shape
            )
            
            # Find best GPU configuration
            gpu_combo, gpu_time, _ = find_best_configuration(
                df, precision, 'GPU', pc, shape
            )
            
            if cpu_time is not None and gpu_time is not None:
                cpu_times.append(cpu_time)
                gpu_times.append(gpu_time)
                cpu_labels.append(cpu_combo)
                
                # GPU label includes speedup
                speedup = cpu_time / gpu_time
                gpu_labels.append(f'{gpu_combo} ({speedup:.1f}x)')
                
                # Y-axis label: particle count with pair count
                y_labels.append(f'{pc}\n({pair_count})')
                
                all_times.extend([cpu_time, gpu_time])
        
        shape_data[shape] = {
            'cpu_times': cpu_times,
            'gpu_times': gpu_times,
            'cpu_labels': cpu_labels,
            'gpu_labels': gpu_labels,
            'y_labels': y_labels
        }
    
    # Compute global x-axis range (log scale with extra space for labels)
    if len(all_times) > 0:
        min_time = min(all_times)
        max_time = max(all_times)
        # Expand range by factor for label space (in log space)
        x_min = min_time * 0.5
        x_max = max_time * 5.0  # Extra space for text labels
    else:
        x_min, x_max = 0.001, 100
    
    for idx, shape in enumerate(shapes):
        ax = axes[idx // 2, idx % 2]
        
        # Get data for this shape
        data = shape_data.get(shape)
        if data is None or len(data['cpu_times']) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(shape_titles.get(shape, shape), fontweight='bold')
            continue
        
        cpu_times = data['cpu_times']
        gpu_times = data['gpu_times']
        cpu_labels = data['cpu_labels']
        gpu_labels = data['gpu_labels']
        y_labels = data['y_labels']
        
        # Create horizontal bar chart
        n = len(cpu_times)
        y_pos = np.arange(n) * 2  # Space out to fit two bars per particle count
        
        # Plot CPU bars first (for legend order) at top position
        bars_cpu = ax.barh(y_pos + 0.9, cpu_times, height=0.8, 
                          color=CPU_COLOR, label='CPU', alpha=0.8)
        
        # Plot GPU bars below CPU
        bars_gpu = ax.barh(y_pos, gpu_times, height=0.8, 
                          color=GPU_COLOR, label='GPU', alpha=0.8)
        
        # Add labels at the end of bars
        for i, (bar_cpu, bar_gpu) in enumerate(zip(bars_cpu, bars_gpu)):
            # CPU label
            width_cpu = bar_cpu.get_width()
            ax.text(width_cpu * 1.15, bar_cpu.get_y() + bar_cpu.get_height()/2,
                   cpu_labels[i], ha='left', va='center', fontsize=12, fontweight='bold')
            
            # GPU label
            width_gpu = bar_gpu.get_width()
            ax.text(width_gpu * 1.15, bar_gpu.get_y() + bar_gpu.get_height()/2,
                   gpu_labels[i], ha='left', va='center', fontsize=12, fontweight='bold')
        
        # Set y-ticks at center of the pair (between CPU and GPU bars)
        ax.set_yticks(y_pos + 0.9)
        ax.set_yticklabels(y_labels, rotation=90, va='center', ha='center')
        # Push y-tick labels slightly outside the plot area for readability
        ax.tick_params(axis='y', which='major', pad=12)
        
        # Labels and title
        ax.set_xlabel(r'Clock Time [ms]', fontweight='bold')
        ax.set_ylabel(r'No. Particles' + '\n' + r'(Pairs)', fontweight='bold')
        ax.set_title(shape_titles.get(shape, shape), fontweight='bold')
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        
        # Grid
        ax.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Legend (only for first subplot)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=14)
    
    plt.tight_layout()
    # Add extra left margin so y-labels are fully visible outside the axes
    plt.subplots_adjust(left=0.16)
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def plot_all_combinations(csv_file, precision, platform, output_file):
    """Create 2x2 subplot showing all 8 GJK combinations for selected particle counts"""
    # Read data
    df = pd.read_csv(csv_file)
    
    # Add shape label and combo code
    df['ShapeLabel'] = df.apply(create_shape_label, axis=1)
    df['ComboCode'] = df.apply(create_combo_code, axis=1)
    
    # Define shape categories
    shapes = ['S1', 'B1', 'SQ4', 'B4']
    shape_titles = {
        'S1': 'Sphere (AR=1)',
        'B1': 'Box (AR=1)',
        'SQ4': 'Superquadric (AR=4)',
        'B4': 'Box (AR=4)'
    }
    
    # Get all 8 combinations in sorted order (algo-rep-trans)
    all_combos = ['JQA', 'JQR', 'JTA', 'JTR', 'SQA', 'SQR', 'STA', 'STR']
    
    # Filter by precision and platform
    df_filtered = df[(df['Precision'] == precision) & (df['Platform'] == platform)].copy()
    
    # Get particle counts and select smallest, median, largest
    all_particle_counts = sorted(df_filtered['ParticleCount'].unique())
    if len(all_particle_counts) == 0:
        print(f"No data for {platform}-{precision}")
        return
    
    if len(all_particle_counts) >= 3:
        selected_counts = [
            all_particle_counts[-1],  # Largest
            all_particle_counts[len(all_particle_counts)//2],  # Median
            all_particle_counts[0]  # Smallest (will be on top)
        ]
    else:
        selected_counts = list(reversed(all_particle_counts))
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'GJK Configuration Comparison - {platform} {precision} Precision', fontsize=18, y=0.995)
    
    # Compute global x-axis range across all data
    all_times = []
    
    for shape in shapes:
        for pc in selected_counts:
            subset = df_filtered[
                (df_filtered['ShapeLabel'] == shape) & 
                (df_filtered['ParticleCount'] == pc)
            ]
            if len(subset) > 0:
                combo_medians = subset.groupby('ComboCode')['TotalTime_ms'].median()
                all_times.extend(combo_medians.values.tolist())
    
    if len(all_times) > 0:
        min_time = min(all_times)
        max_time = max(all_times)
        x_min = min_time * 0.5
        x_max = max_time * 3.0
    else:
        x_min, x_max = 0.001, 100
    
    for idx, shape in enumerate(shapes):
        ax = axes[idx // 2, idx % 2]
        
        # Collect data for this shape
        all_data = []
        y_labels = []
        
        for pc in selected_counts:
            subset = df_filtered[
                (df_filtered['ShapeLabel'] == shape) & 
                (df_filtered['ParticleCount'] == pc)
            ]
            
            if len(subset) == 0:
                continue
            
            # Get pair count
            pair_count = subset['PairCount'].iloc[0]
            y_labels.append(f'{pc}\n({pair_count})')
            
            # Compute median for each combo
            combo_times = []
            for combo in all_combos:
                combo_subset = subset[subset['ComboCode'] == combo]
                if len(combo_subset) > 0:
                    median_time = combo_subset['TotalTime_ms'].median()
                    combo_times.append(median_time)
                else:
                    combo_times.append(np.nan)  # Use NaN for missing data
            
            all_data.append(combo_times)
        
        if len(all_data) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(shape_titles.get(shape, shape), fontweight='bold')
            continue
        
        # Create horizontal grouped bar chart
        n_particles = len(all_data)
        n_combos = len(all_combos)
        bar_height = 0.8 / n_combos  # Divide space among 8 combos
        y_positions = np.arange(n_particles) * (n_combos * bar_height + 0.5)
        
        # Plot bars for each combo
        for combo_idx, combo in enumerate(all_combos):
            combo_values = [all_data[i][combo_idx] for i in range(n_particles)]
            # Replace NaN with 0 for plotting (won't show on log scale)
            combo_values_safe = [v if not np.isnan(v) and v > 0 else 0.001 for v in combo_values]
            y_pos = y_positions + combo_idx * bar_height
            
            ax.barh(y_pos, combo_values_safe, height=bar_height * 0.9,
                   color=COMBO_COLORS[combo_idx], label=combo, alpha=0.9)
        
        # Set y-ticks at center of each particle group
        ax.set_yticks(y_positions + (n_combos * bar_height) / 2 - bar_height / 2)
        ax.set_yticklabels(y_labels, rotation=90, va='center', ha='center')
        ax.tick_params(axis='y', which='major', pad=12)
        
        # Labels and title
        ax.set_xlabel(r'Clock Time [ms]', fontweight='bold')
        ax.set_ylabel(r'No. Particles' + '\n' + r'(Pairs)', fontweight='bold')
        ax.set_title(shape_titles.get(shape, shape), fontweight='bold')
        
        # Set log scale and consistent range
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        
        # Grid
        ax.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Legend (only for first subplot, inside lower right)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Reverse the legend order
            ax.legend(reversed(handles), reversed(labels), loc='lower right', fontsize=10, ncol=1)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.16)
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def plot_benchmarks(csv_file='data/collision_benchmark_comprehensive.csv'):
    """Generate plots for both Single and Double precision"""
    # Check if file exists
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return
    
    # Get unique precisions
    precisions = df['Precision'].unique()
    
    # Generate optimal configuration plots
    print("\n" + "="*80)
    print("Generating Optimal Configuration Plots")
    print("="*80)
    for precision in precisions:
        output_file = f'data/collision_benchmark_{precision.lower()}.eps'
        print(f"\nGenerating plot for {precision} precision...")
        plot_precision(csv_file, precision, output_file)
    
    # Generate all combinations plots for each platform-precision combo
    print("\n" + "="*80)
    print("Generating All Combinations Comparison Plots")
    print("="*80)
    platforms = ['CPU', 'GPU']
    for platform in platforms:
        for precision in precisions:
            output_file = f'data/collision_combos_{platform.lower()}_{precision.lower()}.eps'
            print(f"\nGenerating plot for {platform} {precision} precision...")
            plot_all_combinations(csv_file, precision, platform, output_file)

if __name__ == '__main__':
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/collision_benchmark_comprehensive.csv'
    
    plot_benchmarks(csv_file)
