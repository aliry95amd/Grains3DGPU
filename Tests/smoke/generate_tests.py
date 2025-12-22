#!/usr/bin/env python3
"""
Parametric test generator for GrainsGPU smoke tests.
Generates XML test files from parameter ranges and parameter groups.
"""

import itertools
import json
from pathlib import Path
from typing import Dict, List, Any

# Configuration file paths
CONFIG_FILE = Path(__file__).parent / "test_params.json"
TEMPLATE_FILE = Path(__file__).parent / "template.xml"
OUTPUT_DIR = Path(__file__).parent / "generated"


def load_config() -> Dict[str, Any]:
    """Load test parameter configuration from JSON file."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def expand_param_groups(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand parameter groups where parameters can be arrays (ranges)."""
    expanded_groups = []
    
    for group in param_groups:
        # Separate parameters with ranges from single values
        range_params = {k: v for k, v in group.items() if isinstance(v, list) and k != 'group'}
        fixed_params = {k: v for k, v in group.items() if not isinstance(v, list) or k == 'group'}
        
        if not range_params:
            # No ranges in this group, keep as is
            expanded_groups.append(group)
        else:
            # Generate all combinations of range parameters
            range_names = list(range_params.keys())
            range_values = [range_params[name] for name in range_names]
            
            for values in itertools.product(*range_values):
                new_group = fixed_params.copy()
                new_group.update(dict(zip(range_names, values)))
                expanded_groups.append(new_group)
    
    return expanded_groups


def is_valid_combination(params: Dict[str, Any]) -> bool:
    """Check if a parameter combination is valid."""
    # Skip non-zero sort/update frequencies for BruteForce
    if params.get('neighbor_list_type') == 'BruteForce':
        if params.get('sorting_frequency', 0) != 0 or params.get('update_frequency', 0) != 0:
            return False
    
    return True


def generate_test_name(group_name: str, params: Dict[str, Any], param_names: List[str], base_params: Dict[str, Any]) -> str:
    """Generate a unique test name from group and varying parameters."""
    if group_name:
        # Include both base_params keys (from group) and param_names (from ranges)
        all_varying_params = list(base_params.keys()) + param_names
        param_str = "_".join(f"{k}{v}".replace(" ", "").replace("_", "")
                            for k, v in params.items() 
                            if k in all_varying_params and k not in ['sim_type', 'precision', 'neighbor_list_type'])
        return f"{group_name}_{param_str}" if param_str else group_name
    else:
        return "_".join(f"{k}{v}".replace(" ", "").replace("_", "")
                       for k, v in params.items() if k in param_names)


def generate_linked_cell_config(params: Dict[str, Any]) -> str:
    """Generate LinkedCell XML configuration string."""
    if params.get('neighbor_list_type') != 'LinkedCell':
        return ''
    
    lc_type = params.get('linked_cell_type', 'Host')
    sort_freq = params.get('sorting_frequency', 0)
    update_freq = params.get('update_frequency', 0)
    cell_size_factor = params.get('cell_size_factor', 1.0)
    
    return (f'<LinkedCell Type="{lc_type}" '
            f'CellSizeFactor="{cell_size_factor}" UpdatingFrequency="{update_freq}" '
            f'SortingFrequency="{sort_freq}"/>')


def generate_particles_section(params: Dict[str, Any]) -> str:
    """Generate particles XML section with all shapes distributed equally."""
    particle_shapes = params.get('particle_shapes', ['Sphere'])
    num_particles = params.get('num_particles', 50)
    particles_per_shape = num_particles // len(particle_shapes)
    density = params.get('particle_density', 1000)
    crust = params.get('crust_thickness', 0.0002)
    
    # Shape geometry definitions
    geometries = {
        'Sphere': '<Sphere Radius="0.08"/>',
        'Box': '<Box LX="0.0924" LY="0.0924" LZ="0.0924"/>',
        'Cylinder': '<Cylinder Radius="0.0566" Height="0.1131"/>',
        'Superquadric': '<Superquadric a="0.01" b="0.01" c="0.08" n1="2.0" n2="2.0"/>'
    }
    
    particles_section = []
    for shape in particle_shapes:
        geometry = geometries.get(shape, '<Sphere Radius="0.08"/>')
        particle_entry = f'''		<Particle Number="{particles_per_shape}" Density="{density}" Material="mat1">
			<Convex CrustThickness="{crust}">
				{geometry}
			</Convex>
		</Particle>'''
        particles_section.append(particle_entry)
    
    return '\n'.join(particles_section)


def generate_tests(config: Dict[str, Any]) -> None:
    """Generate XML test files from parameter ranges."""
    # Load template
    template = TEMPLATE_FILE.read_text()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get configurations
    param_groups = config.get('param_groups', [])
    param_ranges = config.get('param_ranges', {})
    fixed_params = config.get('fixed_params', {})
    
    # Expand groups with parameter ranges
    expanded_groups = expand_param_groups(param_groups)
    
    test_count = 0
    
    # Generate tests for each parameter group
    for test_config in expanded_groups:
        # Extract base parameters from config
        group_name = test_config.get('group', '')
        base_params = {k: v for k, v in test_config.items() if k != 'group'}
        
        # Generate all combinations of remaining parameters
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        for values in itertools.product(*param_values):
            # Create parameter dictionary
            params = base_params.copy()
            params.update(dict(zip(param_names, values)))
            params.update(fixed_params)
            
            # Skip invalid combinations
            if not is_valid_combination(params):
                continue
            
            # Generate test name
            test_name = generate_test_name(group_name, params, param_names, base_params)
            params['test_name'] = test_name
            
            # Generate configuration strings
            params['linked_cell_config'] = generate_linked_cell_config(params)
            params['domain_center'] = params.get('domain_size', 1.152) / 2.0
            params['particles_section'] = generate_particles_section(params)
            
            # Format template
            try:
                xml_content = template.format(**params)
            except KeyError as e:
                print(f"Error: Missing parameter {e} in template")
                continue
            
            # Write XML file
            output_file = OUTPUT_DIR / f"{test_name}.xml"
            output_file.write_text(xml_content)
            test_count += 1
            print(f"Generated: {output_file.name}")
    
    print(f"\nTotal tests generated: {test_count}")
    
    # Generate test list file
    test_list = OUTPUT_DIR / "test_list.txt"
    test_list.write_text("\n".join(
        f.stem for f in sorted(OUTPUT_DIR.glob("*.xml"))
    ))
    print(f"Test list written to: {test_list}")


def main():
    """Main entry point."""
    if not CONFIG_FILE.exists():
        print(f"Error: Configuration file not found: {CONFIG_FILE}")
        print("Please create test_params.json with parameter ranges")
        return 1
    
    if not TEMPLATE_FILE.exists():
        print(f"Error: Template file not found: {TEMPLATE_FILE}")
        return 1
    
    config = load_config()
    generate_tests(config)
    return 0


if __name__ == "__main__":
    exit(main())
