#!/usr/bin/env python3
"""Generate Grains3D static-packing insert.xml (box obstacles, particles, insertion)."""

import argparse
from cmath import acos, cos, sqrt
import os
from typing import Dict, Any

def generate_particle_block(config: Dict[str, Any]) -> str:
    """XML <Particle> block for the configured convex shape."""
    particle_type = config.get('particle_type', 'S1')
    num_particles = config['n']
    r = config['r']
    density = config.get('density', 1000)
    crust_thickness = r * 0.01

    if particle_type == 'S1':
        return f'''		<Particle Number="{num_particles}" Density="{density}" Material="matP">
			<Convex CrustThickness="{crust_thickness}">
				<Sphere Radius="{r}"/>
			</Convex>
		</Particle>'''

    if particle_type == 'S4':
        a = r/2
        b = r/2
        c = 4*r
        n1 = 2.0
        n2 = 2.0
        return f'''		<Particle Number="{num_particles}" Density="{density}" Material="matP">
			<Convex CrustThickness="{crust_thickness}">
				<Superquadric a="{a}" b="{b}" c="{c}" n1="{n1}" n2="{n2}"/>
			</Convex>
		</Particle>'''
    
    elif particle_type == 'B1':
        lx = 2/sqrt(3) * r
        ly = lx
        lz = lx
        return f'''		<Particle Number="{num_particles}" Density="{density}" Material="matP">
			<Convex CrustThickness="{crust_thickness}">
				<Box LX="{lx}" LY="{ly}" LZ="{lz}"/>
			</Convex>
		</Particle>'''
    
    
    elif particle_type == 'B4':
        A = cos(1/3*acos(-1/64))
        lx = 1/sqrt(6*A) * r
        ly = lx
        lz = 16 / sqrt(3) * A * r
        return f'''		<Particle Number="{num_particles}" Density="{density}" Material="matP">
			<Convex CrustThickness="{crust_thickness}">
				<Box LX="{lx}" LY="{ly}" LZ="{lz}"/>
			</Convex>
		</Particle>'''

def generate_xml(config: Dict[str, Any]) -> str:
    """Full Grains3D XML string for `config`."""
    half_width = config['L'] / 2.0
    half_height = config['H'] / 2.0

    domain_x = config['L']
    domain_y = config['L']
    domain_z = config['H']

    insertion_min_x = 0.0
    insertion_min_y = 0.0
    insertion_min_z = 0.0
    insertion_max_x = domain_x
    insertion_max_y = domain_y
    insertion_max_z = domain_z

    xml_content = f'''<Grains3D Type="{"Standard"}" Precision="{config['precision']}">

<Construction>
	<Origin X="0.0" Y="0.0" Z="0.0"/>
	<MaxCoordinate X="{domain_x}" Y="{domain_y}" Z="{domain_z}"/>
	<Periodicity PX="0" PY="0" PZ="0"/>

	<CollisionDetection>
		<NeighborList Type="{config['neighbor_list_type']}" UpdateFrequency="{config['update_frequency']}"/>
		<LinkedCell>
			<Type>DeviceMemoryEfficient</Type>
			<CellOrdering>Morton</CellOrdering>
			<CellSizeFactor>1.</CellSizeFactor>
			<UpdatingFrequency>0</UpdatingFrequency>
			<SortingFrequency>0</SortingFrequency>
		</LinkedCell>
		<BoundingVolume Type="OBB"/>
		<NarrowPhase Type="GJK"/>
	</CollisionDetection>

	<ContactForceModels>
		<ContactForceModel Type='Hooke'>
			<Material materialA="matP" materialB="matP"/>
			<Parameters>
				<kn>{config['kn']}</kn>
				<en>{config['en']}</en>
				<etat>{config['etat']}</etat>
				<muc>{config['muc']}</muc>
				<kr>{config['kr']}</kr>
			</Parameters>
		</ContactForceModel>
	</ContactForceModels>

	<TemporalSetting>
		<TimeInterval Start="0." End="{config['end_time']}" dt="{config['dt']}"/>
		<TimeIntegration Type="FirstOrderExplicit"/>
	</TemporalSetting>

	<Particles>
{generate_particle_block(config)}
	</Particles>

	<Obstacles>
		<Obstacle Material="matP">
			<Convex CrustThickness="1.e-5">
				<Rectangle LX="{config['width']}" LY="{config['depth']}" LZ="0.0"/>
			</Convex>
			<Transformation>
				<Centre X="{half_width}" Y="{half_width}" Z="0.0"/>
				<AngularPosition Type="Angle" aX="0." aY="0." aZ="0"/>
			</Transformation>
		</Obstacle>
		<Obstacle Material="matP">
			<Convex CrustThickness="1.e-5">
				<Rectangle LX="{config['height']}" LY="{config['depth']}" LZ="0.001"/>
			</Convex>
			<Transformation>
				<Centre X="{config['width']}" Y="{half_width}" Z="{half_height}"/>
				<AngularPosition Type="Angle" aX="0." aY="-90." aZ="0."/>
			</Transformation>
		</Obstacle>
		<Obstacle Material="matP">
			<Convex CrustThickness="1.e-5">
				<Rectangle LX="{config['height']}" LY="{config['depth']}" LZ="0.001"/>
			</Convex>
			<Transformation>
				<Centre X="0.0" Y="{half_width}" Z="{half_height}"/>
				<AngularPosition Type="Angle" aX="0." aY="90." aZ="0."/>
			</Transformation>
		</Obstacle>
		<Obstacle Material="matP">
			<Convex CrustThickness="1.e-5">
				<Rectangle LX="{config['width']}" LY="{config['height']}" LZ="0.001"/>
			</Convex>
			<Transformation>
				<Centre X="{half_width}" Y="{config['depth']}" Z="{half_height}"/>
				<AngularPosition Type="Angle" aX="90." aY="0." aZ="0."/>
			</Transformation>
		</Obstacle>
		<Obstacle Material="matP">
			<Convex CrustThickness="1.e-5">
				<Rectangle LX="{config['width']}" LY="{config['height']}" LZ="0.001"/>
			</Convex>
			<Transformation>
				<Centre X="{half_width}" Y="0.0" Z="{half_height}"/>
				<AngularPosition Type="Angle" aX="-90." aY="0." aZ="0."/>
			</Transformation>
		</Obstacle>
	</Obstacles>
</Construction>

<Forces>
	<Gravity GX="0.0" GY="0.0" GZ="{config['gravity']}"/>
</Forces>

<Simulation>
	<ParticleInsertion ForceInsertion="0">
		<InitialPosition Type="Random" Seed="UserDefined" Value="{config['random_seed']}">
			<Windows>
				<Window Type="Box">
					<MinPoint X="{insertion_min_x}" Y="{insertion_min_y}" Z="{insertion_min_z}"/>
					<MaxPoint X="{insertion_max_x}" Y="{insertion_max_y}" Z="{insertion_max_z}"/>
				</Window>
			</Windows>
		</InitialPosition>
        <InitialOrientation Type="Random" Seed="UserDefined" Value="{config['random_seed']}">
			<Windows>
				<Window Type="Box">
					<MinPoint X="{0}" Y="{0}" Z="{0}"/>
					<MaxPoint X="{6.28}" Y="{6.28}" Z="{6.28}"/>
				</Window>
			</Windows>
		</InitialOrientation>
	</ParticleInsertion>
	<PostProcessing>
		<TimeSave dt="{config['save_dt']}"/>
		<Writers>
			<Paraview Directory="./Grains/Init" RootName="{config['output_name']}"/>
			<RawData  Directory="./Grains/Init" RootName="{config['output_name']}"/>
		</Writers>
	</PostProcessing>
</Simulation>

</Grains3D>'''
    
    return xml_content

def get_default_config() -> Dict[str, Any]:
    """Defaults for static packing XML generation."""
    return {
        'type': 'GPU',
        'precision': 'Double',

        'L': 1.0,
        'H': 5.0,

        'n': 200,
        'r': 0.04,
        'particle_type': 'S1',
        'density': 1000,

        'neighbor_list_type': 'LinkedCell',
        'update_frequency': 1,

        'kn': 1.2e8,
        'en': 0.1,
        'etat': 1.0e2,
        'muc': 0.5,
        'kr': 0.0,

        'dt': 1.0e-5,
        'end_time': 1.0e-5,
        'save_dt': 1.0e-5,

        'gravity': -9.81,

        'random_seed': 1,

        'output_name': 'insert',
    }

def main():
    parser = argparse.ArgumentParser(description='Generate XML for Grains3D static packing simulation')

    parser.add_argument('--L', type=float, help='Box width and depth (X and Y direction)')
    parser.add_argument('--H', type=float, help='Box height (Z direction)')

    parser.add_argument('--type', '--t', choices=['S1', 'S4', 'B1', 'B4'], 
                       default='S1', help='Particle type')
    parser.add_argument('--n', type=int, help='Number of particles')
    parser.add_argument('--r', type=float, help='Particle radius (or characteristic size)')    

    parser.add_argument('--output', '-o', default='insert.xml', help='Output XML filename')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output messages')
    
    args = parser.parse_args()

    config = get_default_config()

    if args.L is not None:
        config['L'] = args.L
        config['width'] = args.L
        config['depth'] = args.L
    if args.H is not None:
        config['H'] = args.H
        config['height'] = args.H
    if args.type is not None:
        config['particle_type'] = args.type
    if args.n is not None:
        config['n'] = args.n
    if args.r is not None:
        config['r'] = args.r
        config['particle_radius'] = args.r
    if args.output is not None:
        config['output_name'] = args.output

    xml_content = generate_xml(config)

    with open(f"{config['output_name']}.xml", 'w') as f:
        f.write(xml_content)
    
    if not args.quiet:
        print(f"Generated {config['output_name']}.xml with:")
        print(f"  Box: {config['L']} x {config['L']} x {config['H']}")
        print(f"  Particles: {config['n']} {config['particle_type']} particles")
        if config['particle_type'] == 'sphere':
            print(f"    Radius: {config['particle_radius']}")
        elif config['particle_type'] == 'box':
            print(f"    Dimensions: {config['box_lx']} x {config['box_ly']} x {config['box_lz']}")
        elif config['particle_type'] == 'cylinder':
            print(f"    Radius: {config['particle_radius']}, Height: {config['cylinder_height']}")
        elif config['particle_type'] == 'superquadric':
            print(f"    Parameters: a={config['sq_a']}, b={config['sq_b']}, c={config['sq_c']}, n1={config['sq_n1']}, n2={config['sq_n2']}")
        print(f"  Density: {config['density']}")
    
if __name__ == '__main__':
    main()