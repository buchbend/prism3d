#!/usr/bin/env python3
"""
PRISM-3D HPC Runner — command-line entry point.

Usage:
  python -m prism3d.run --n 64 --G0 1000 --box 2.0 --output ./output
  python -m prism3d.run --config model.json
  python -m prism3d.run --orion-bar --n 64

See docs/RUNNING.md for full documentation.
"""

import argparse
import json
import os
import sys
import time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='PRISM-3D: 3D PDR code with THEMIS dust evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (8³, ~20s)
  python -m prism3d.run --n 8 --G0 100

  # Science run (64³, needs HPC)
  python -m prism3d.run --n 64 --G0 1000 --box 5.0 --n-mean 300 --output ./run1

  # Orion Bar model
  python -m prism3d.run --orion-bar --n 32

  # From config file
  python -m prism3d.run --config my_model.json

  # Generate SLURM script
  python -m prism3d.run --gen-slurm --n 128 --nodes 4
        """)

    # Grid parameters
    p.add_argument('--n', type=int, default=8, help='Cells per dimension (default: 8)')
    p.add_argument('--box', type=float, default=2.0, help='Box size [pc] (default: 2.0)')
    p.add_argument('--n-mean', type=float, default=500, help='Mean density [cm⁻³] (default: 500)')
    p.add_argument('--sigma-ln', type=float, default=1.5, help='Log-normal width (default: 1.5)')

    # Physics
    p.add_argument('--G0', type=float, default=100, help='External FUV field [Habing] (default: 100)')
    p.add_argument('--zeta-cr', type=float, default=2e-16, help='CR ionization rate [s⁻¹]')
    p.add_argument('--nside', type=int, default=1, help='HEALPix nside for RT (default: 1)')

    # Solver
    p.add_argument('--max-iter', type=int, default=10, help='Max iterations (default: 10)')
    p.add_argument('--tol', type=float, default=0.05, help='Convergence tolerance')
    p.add_argument('--dust-steps', type=int, default=5, help='Outer dust evolution steps (default: 5)')
    p.add_argument('--refine', action='store_true', help='Run BDF refinement pass')
    p.add_argument('--accelerator', type=str, default=None,
                   help='Path to trained ML accelerator .pkl file')

    # I/O
    p.add_argument('--output', type=str, default='./prism3d_output', help='Output directory')
    p.add_argument('--config', type=str, help='JSON config file (overrides CLI args)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')

    # Presets
    p.add_argument('--orion-bar', action='store_true', help='Use Orion Bar preset')
    p.add_argument('--horsehead', action='store_true', help='Use Horsehead preset')

    # HPC
    p.add_argument('--gen-slurm', action='store_true', help='Generate SLURM script and exit')
    p.add_argument('--nodes', type=int, default=1, help='Number of nodes for SLURM')
    p.add_argument('--partition', type=str, default='gpu', help='SLURM partition')
    p.add_argument('--wall-time', type=str, default='04:00:00', help='Wall time')

    return p.parse_args()


def run_model(args):
    """Run a PRISM-3D model from parsed arguments."""
    # Defer imports so --help is fast
    from prism3d.density_fields import fractal_turbulent
    from prism3d.solver_3d import PDRSolver3D
    from prism3d.observations.jwst_pipeline import generate_observations, plot_observations
    from prism3d.utils.constants import pc_cm

    # Apply presets
    if args.orion_bar:
        args.G0 = 20000
        args.n_mean = 50000
        args.box = 0.4
        args.sigma_ln = 0.8
        args.nside = 2
        print("Using Orion Bar preset: G0=2e4, n=5e4, box=0.4 pc")
    elif args.horsehead:
        args.G0 = 100
        args.n_mean = 2000
        args.box = 0.5
        args.sigma_ln = 1.0
        print("Using Horsehead preset: G0=100, n=2e3, box=0.5 pc")

    # Load config file if given
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k.replace('-', '_')):
                setattr(args, k.replace('-', '_'), v)
        print(f"Loaded config from {args.config}")

    n = args.n
    box = args.box * pc_cm

    # Create timestamped run directory inside the output dir
    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.output = run_dir

    print(f"\n{'='*60}")
    print(f"  PRISM-3D v0.5 — 3D PDR with THEMIS Dust Evolution")
    print(f"{'='*60}")
    print(f"  Grid:    {n}³ = {n**3:,} cells")
    print(f"  Box:     {args.box:.2f} pc ({args.box*1e3:.0f} mpc resolution)")
    print(f"  <n_H>:   {args.n_mean:.0f} cm⁻³")
    print(f"  G₀:      {args.G0:.0f} Habing")
    print(f"  ζ_CR:    {args.zeta_cr:.1e} s⁻¹")
    print(f"  RT:      nside={args.nside} ({12*args.nside**2} rays)")
    print(f"  Output:  {args.output}")
    print(f"{'='*60}\n")

    # Generate density field
    density, cs = fractal_turbulent(
        n, box, n_mean=args.n_mean,
        sigma_ln=args.sigma_ln, seed=args.seed
    )

    # Create and run solver
    t0 = time.time()
    solver = PDRSolver3D(
        density, box,
        G0_external=args.G0,
        zeta_CR_0=args.zeta_cr,
        nside_rt=args.nside
    )
    if args.accelerator:
        from prism3d.chemistry.accelerator import ChemistryAccelerator
        solver.accelerator = ChemistryAccelerator.load(args.accelerator)
        print(f"ML accelerator loaded: {args.accelerator}")

    solver.run(max_iterations=args.max_iter,
               convergence_tol=args.tol,
               dust_steps=args.dust_steps, verbose=True)

    if args.refine:
        solver.refine(verbose=True)

    wall = time.time() - t0
    print(f"\nTotal wall time: {wall:.1f}s ({wall/60:.1f} min)")

    # Run full evaluation suite (saves model, generates observations,
    # creates all figures, writes HTML report)
    from prism3d.evaluate import evaluate_model
    report = evaluate_model(solver, output_dir=args.output, verbose=True)

    return solver, report


def generate_slurm(args):
    """Generate a SLURM submission script."""
    script = f"""#!/bin/bash
#SBATCH --job-name=prism3d_{args.n}cube
#SBATCH --output=prism3d_%j.out
#SBATCH --error=prism3d_%j.err
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={max(4, 64 // args.nodes)}
#SBATCH --time={args.wall_time}
#SBATCH --partition={args.partition}

# Load environment
source $HOME/prism3d_env/bin/activate

echo "PRISM-3D starting on $(hostname) at $(date)"
echo "Grid: {args.n}³, G0={args.G0}, box={args.box} pc"

python -m prism3d.run \\
    --n {args.n} \\
    --G0 {args.G0} \\
    --box {args.box} \\
    --n-mean {args.n_mean} \\
    --nside {args.nside} \\
    --max-iter {args.max_iter} \\
    --output {args.output} \\
    {'--refine' if args.refine else ''}

echo "Finished at $(date)"
"""
    path = f'run_prism3d_{args.n}cube.slurm'
    with open(path, 'w') as f:
        f.write(script)
    print(f"SLURM script written: {path}")
    print(f"Submit with: sbatch {path}")


def main():
    args = parse_args()

    if args.gen_slurm:
        generate_slurm(args)
        return

    run_model(args)


if __name__ == '__main__':
    main()
