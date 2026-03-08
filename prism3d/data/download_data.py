#!/usr/bin/env python3
"""
PRISM-3D Data Downloader — PDRs4All Orion Bar Dataset.

Downloads and assembles the complete Orion Bar benchmark dataset
from public archives for testing PRISM-3D against JWST observations.

Data sources:
  1. MAST Archive: JWST Program 1288 (PDRs4All ERS)
     - NIRSpec IFU cubes (G140H, G235H, G395H): 0.97-5.27 μm
     - MIRI MRS cubes (Channels 1-4): 4.9-27.9 μm
     - NIRCam + MIRI imaging
  2. PDRs4All website: Template spectra for 5 PDR zones
  3. VizieR: MIRI MRS template spectra (Chown+ 2024)
  4. Herschel Science Archive: PACS 70/160 μm, SPIRE 250/350/500 μm
  5. ALMA Archive: HCO+ J=4-3, CO J=3-2 (Project 2012.1.00352.S)

Usage:
  # Download everything (requires ~10 GB disk space)
  python download_data.py --all --output ./orion_bar_data

  # Download only template spectra (~10 MB, quick start)
  python download_data.py --templates --output ./orion_bar_data

  # Download only JWST Level 3 cubes (~5 GB)
  python download_data.py --jwst-cubes --output ./orion_bar_data

  # Download Herschel FIR maps (~500 MB)
  python download_data.py --herschel --output ./orion_bar_data

  # Check what's already downloaded
  python download_data.py --check --output ./orion_bar_data

Requirements:
  pip install astropy astroquery spectral-cube requests
"""

import argparse
import os
import sys
import json
import hashlib
from pathlib import Path

# Orion Bar coordinates
ORION_BAR_RA = 83.8375    # deg (05h 35m 21s)
ORION_BAR_DEC = -5.4125   # deg (-05d 24m 45s)
ORION_BAR_RADIUS = 0.03   # deg (~2 arcmin search radius)

# JWST Program IDs
PDRS4ALL_PID = '1288'
HORSEHEAD_PID = '1192'

# Known data products and their URLs
TEMPLATE_URLS = {
    'nirspec_templates': {
        'url': 'https://www.pdrs4all.org/wp-content/uploads/2023/11/nirspec_templates.tar.gz',
        'description': 'NIRSpec IFU template spectra for 5 PDR zones (0.97-5.27 μm)',
        'size_mb': 7,
    },
}

# Herschel observation IDs for the Orion Bar region
HERSCHEL_OBSIDS = {
    'PACS_70': '1342218927',   # PACS 70 μm photometry
    'PACS_160': '1342218927',  # PACS 160 μm (same obsid, dual-band)
    'SPIRE_250': '1342218738', # SPIRE 250 μm
}

# Physical parameters (for reference / model setup)
ORION_BAR_PARAMS = {
    'distance_pc': 414,
    'G0_range': [22000, 71000],  # Habing units
    'G0_best': 26000,
    'n_H_range': [50000, 100000],  # cm^-3
    'n_H_clumps': 1e6,
    'T_gas_surface': 500,  # K (PDR surface)
    'T_gas_deep': 30,      # K (molecular zone)
    'T_dust_range': [35, 70],  # K
    'pressure_range': [3e7, 9e7],  # K cm^-3
    'inclination_deg': 4,  # Nearly edge-on
    'theta1_ori_c': {
        'spectral_type': 'O7V',
        'T_eff': 39000,
        'distance_to_bar_pc': 0.33,
    },
    'dissociation_fronts': {
        'DF1_pc': 0.250,  # Distance from star
        'DF2_pc': 0.257,
        'DF3_pc': 0.267,
    },
}


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg in ['astropy', 'requests']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Optional but recommended
    optional_missing = []
    for pkg in ['astroquery', 'spectral_cube']:
        try:
            __import__(pkg)
        except ImportError:
            optional_missing.append(pkg)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    if optional_missing:
        print(f"WARNING: Optional packages not installed: {', '.join(optional_missing)}")
        print(f"  Install for full functionality: pip install {' '.join(optional_missing)}")
        print()

    return len(optional_missing) == 0


def download_file(url, filepath, description=""):
    """Download a file with progress bar."""
    import requests

    if os.path.exists(filepath):
        print(f"  ✓ Already exists: {filepath}")
        return True

    print(f"  ↓ Downloading: {description or url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r    {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)",
                          end='', flush=True)
        print(f"\r    ✓ Saved: {filepath} ({os.path.getsize(filepath)/1e6:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n    ✗ Failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def download_templates(output_dir):
    """Download template spectra from PDRs4All website."""
    print("\n" + "="*60)
    print("  Downloading PDRs4All Template Spectra")
    print("="*60)

    template_dir = os.path.join(output_dir, 'templates')
    os.makedirs(template_dir, exist_ok=True)

    for name, info in TEMPLATE_URLS.items():
        filepath = os.path.join(template_dir, f'{name}.tar.gz')
        success = download_file(info['url'], filepath, info['description'])

        if success and filepath.endswith('.tar.gz'):
            import tarfile
            try:
                with tarfile.open(filepath, 'r:gz') as tar:
                    tar.extractall(path=template_dir)
                print(f"    Extracted to {template_dir}")
            except Exception as e:
                print(f"    Extract failed: {e}")

    # Write a README for the templates
    readme = """# PDRs4All Template Spectra

Five template spectra extracted from the JWST NIRSpec and MIRI MRS
observations of the Orion Bar, corresponding to five distinct PDR zones:

1. **HII**    — Ionized gas (foreground H II region)
2. **Atomic** — Atomic PDR layer (between IF and DF1)
3. **DF1**    — First dissociation front
4. **DF2**    — Second dissociation front
5. **DF3**    — Third dissociation front (deepest)

## Loading in Python

```python
from astropy.io import fits
from astropy.table import Table

# NIRSpec template
spec = Table.read('nirspec_template_DF1.fits')
wavelength = spec['WAVELENGTH']  # μm
flux = spec['FLUX']              # MJy/sr

# Or from the tar.gz:
import tarfile
with tarfile.open('nirspec_templates.tar.gz') as tar:
    tar.extractall()
```

## Reference
Peeters et al. 2024, A&A 685, A74 (NIRSpec)
Chown et al. 2024, A&A 685, A75 (MIRI MRS)
"""
    with open(os.path.join(template_dir, 'README.md'), 'w') as f:
        f.write(readme)


def download_jwst_cubes(output_dir):
    """Download JWST Level 3 IFU cubes from MAST."""
    print("\n" + "="*60)
    print("  Downloading JWST IFU Cubes (Program 1288)")
    print("="*60)

    cube_dir = os.path.join(output_dir, 'jwst_cubes')
    os.makedirs(cube_dir, exist_ok=True)

    try:
        from astroquery.mast import Observations

        print("  Querying MAST for Program 1288...")
        obs_table = Observations.query_criteria(
            proposal_id=PDRS4ALL_PID,
            obs_collection='JWST',
            dataRights='PUBLIC',
        )
        print(f"  Found {len(obs_table)} observations")

        # Filter for NIRSpec and MIRI MRS IFU observations
        ifu_mask = (
            (obs_table['instrument_name'] == 'NIRSPEC/IFU') |
            (obs_table['instrument_name'] == 'MIRI/MRS')
        )
        ifu_obs = obs_table[ifu_mask]
        print(f"  IFU observations: {len(ifu_obs)}")

        if len(ifu_obs) == 0:
            print("  No IFU observations found. Check network connectivity.")
            _write_manual_instructions(cube_dir)
            return

        # Get data products
        products = Observations.get_product_list(ifu_obs)

        # Filter for Level 3 science cubes (s3d)
        s3d_products = Observations.filter_products(
            products,
            calib_level=[3],
            productType='SCIENCE',
            extension='fits',
        )

        # Further filter for s3d files
        s3d_mask = ['s3d' in str(p).lower() for p in s3d_products['productFilename']]
        s3d_final = s3d_products[s3d_mask]

        if len(s3d_final) == 0:
            # Try without s3d filter
            s3d_final = s3d_products[:20]  # Take first 20

        print(f"  Downloading {len(s3d_final)} Level 3 cubes...")
        manifest = Observations.download_products(
            s3d_final,
            download_dir=cube_dir,
        )
        print(f"  Downloaded {len(manifest)} files")

    except ImportError:
        print("  astroquery not installed. Writing manual download instructions.")
        _write_manual_instructions(cube_dir)
    except Exception as e:
        print(f"  Download failed: {e}")
        _write_manual_instructions(cube_dir)


def _write_manual_instructions(output_dir):
    """Write manual download instructions if automated download fails."""
    instructions = """# Manual JWST Data Download Instructions

If the automated download failed, follow these steps:

## Option 1: MAST Portal (browser)

1. Go to https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
2. In "Search" box: enter "1288" in the Proposal ID field
3. Click "Search"
4. Filter: instrument_name = NIRSPEC/IFU or MIRI/MRS
5. Select the observations you want
6. Click "Download" → "Minimum Recommended Products" or "Science Products"
7. Choose Level 3 (combined/calibrated) products

## Option 2: Python (astroquery)

```python
from astroquery.mast import Observations

# Query
obs = Observations.query_criteria(proposal_id='1288', obs_collection='JWST')

# Get NIRSpec IFU products
nirspec = obs[obs['instrument_name'] == 'NIRSPEC/IFU']
products = Observations.get_product_list(nirspec)
s3d = Observations.filter_products(products, calib_level=[3])

# Download
Observations.download_products(s3d, download_dir='./jwst_cubes')
```

## Option 3: Direct URL (if you know the filename)

Level 3 products follow the pattern:
  https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/<filename>

## Key files to look for:
  - jw01288*nirspec*g140h*s3d.fits  (NIRSpec 0.97-1.89 μm)
  - jw01288*nirspec*g235h*s3d.fits  (NIRSpec 1.66-3.17 μm)
  - jw01288*nirspec*g395h*s3d.fits  (NIRSpec 2.87-5.27 μm)
  - jw01288*miri*ch1*s3d.fits       (MIRI MRS 4.9-7.76 μm)
  - jw01288*miri*ch2*s3d.fits       (MIRI MRS 7.45-11.87 μm)
  - jw01288*miri*ch3*s3d.fits       (MIRI MRS 11.47-18.24 μm)
  - jw01288*miri*ch4*s3d.fits       (MIRI MRS 17.54-27.9 μm)
"""
    with open(os.path.join(output_dir, 'DOWNLOAD_INSTRUCTIONS.md'), 'w') as f:
        f.write(instructions)
    print(f"  Instructions written to {output_dir}/DOWNLOAD_INSTRUCTIONS.md")


def download_herschel(output_dir):
    """Download Herschel PACS/SPIRE maps of the Orion Bar region."""
    print("\n" + "="*60)
    print("  Downloading Herschel FIR Maps")
    print("="*60)

    herschel_dir = os.path.join(output_dir, 'herschel')
    os.makedirs(herschel_dir, exist_ok=True)

    try:
        from astroquery.esasky import ESASky

        print("  Querying ESASky for Herschel observations...")
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ORION_BAR_RA*u.deg, dec=ORION_BAR_DEC*u.deg)

        # Search for Herschel observations
        maps = ESASky.query_region_maps(
            coord, radius=3*u.arcmin,
            missions=['Herschel']
        )

        if 'HERSCHEL' in maps and len(maps['HERSCHEL']) > 0:
            print(f"  Found {len(maps['HERSCHEL'])} Herschel observations")
            # Download FITS maps
            results = ESASky.get_maps(
                coord, radius=3*u.arcmin,
                missions=['Herschel'],
                download_dir=herschel_dir
            )
            print(f"  Downloaded Herschel maps to {herschel_dir}")
        else:
            print("  No Herschel maps found via ESASky. Try HSA directly.")
            _write_herschel_instructions(herschel_dir)

    except ImportError:
        print("  astroquery.esasky not available.")
        _write_herschel_instructions(herschel_dir)
    except Exception as e:
        print(f"  Download failed: {e}")
        _write_herschel_instructions(herschel_dir)


def _write_herschel_instructions(output_dir):
    """Write manual Herschel download instructions."""
    instructions = """# Herschel Data Download

## Herschel Science Archive
1. Go to http://archives.esac.esa.int/hsa/whsa/
2. Search by coordinates: RA=83.8375, Dec=-5.4125, radius=3 arcmin
3. Filter by instrument: PACS photometer, SPIRE photometer
4. Download Level 2.5 or Level 3 products (fully reduced maps)

## Key Observation IDs:
  - PACS 70/160 μm: 1342218927 (OT1 program)
  - SPIRE 250/350/500 μm: 1342218738

## Alternative: Use published maps
  - Arab et al. 2012, A&A 541, A19 (Herschel dust modeling)
  - Bernard-Salas et al. 2012 (PACS spectroscopy)
"""
    with open(os.path.join(output_dir, 'DOWNLOAD_INSTRUCTIONS.md'), 'w') as f:
        f.write(instructions)


def create_prism3d_config(output_dir):
    """Create a PRISM-3D configuration file for the Orion Bar."""
    print("\n" + "="*60)
    print("  Creating PRISM-3D Orion Bar Configuration")
    print("="*60)

    config = {
        "name": "Orion Bar (PDRs4All)",
        "n": 32,
        "G0": 26000,
        "box": 0.4,
        "n_mean": 50000,
        "sigma_ln": 0.8,
        "zeta_cr": 1e-16,
        "nside": 2,
        "max_iter": 10,
        "tol": 0.05,
        "refine": True,
        "output": "./orion_bar_run",
        "distance_pc": 414,
        "notes": "Based on PDRs4All ERS Program 1288 parameters"
    }

    config_path = os.path.join(output_dir, 'orion_bar_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")
    print(f"  Run with: python -m prism3d.run --config {config_path}")

    # Also write the physical parameters reference
    params_path = os.path.join(output_dir, 'orion_bar_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(ORION_BAR_PARAMS, f, indent=2, default=str)
    print(f"  Physical parameters: {params_path}")


def create_loading_script(output_dir):
    """Create a Python script showing how to load and inspect the data."""
    script = '''#!/usr/bin/env python3
"""
Load and inspect the PDRs4All Orion Bar dataset.

Run after download_data.py to verify the data and produce
quick-look plots.
"""

import os
import glob
import numpy as np

# Check what we have
data_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Data directory: {data_dir}")
print()

# 1. Template spectra
template_dir = os.path.join(data_dir, 'templates')
if os.path.exists(template_dir):
    fits_files = glob.glob(os.path.join(template_dir, '**/*.fits'), recursive=True)
    dat_files = glob.glob(os.path.join(template_dir, '**/*.dat'), recursive=True)
    print(f"Templates: {len(fits_files)} FITS + {len(dat_files)} .dat files")

    if fits_files:
        try:
            from astropy.io import fits
            from astropy.table import Table

            for f in sorted(fits_files)[:3]:
                hdu = fits.open(f)
                print(f"  {os.path.basename(f)}: {len(hdu)} HDUs, "
                      f"{hdu[0].header.get('NAXIS', '?')} axes")
                hdu.close()
        except ImportError:
            print("  (Install astropy to inspect FITS files)")
else:
    print("Templates: not downloaded yet")

# 2. JWST cubes
cube_dir = os.path.join(data_dir, 'jwst_cubes')
if os.path.exists(cube_dir):
    cubes = glob.glob(os.path.join(cube_dir, '**/*s3d*.fits'), recursive=True)
    print(f"\\nJWST IFU cubes: {len(cubes)} files")
    total_size = sum(os.path.getsize(f) for f in cubes) / 1e9
    print(f"  Total size: {total_size:.1f} GB")

    if cubes:
        try:
            from astropy.io import fits

            for cube_path in sorted(cubes)[:5]:
                hdu = fits.open(cube_path)
                sci = hdu['SCI'] if 'SCI' in hdu else hdu[1]
                shape = sci.data.shape if sci.data is not None else 'None'
                name = os.path.basename(cube_path)
                print(f"  {name}: shape={shape}")
                hdu.close()
        except Exception as e:
            print(f"  (Could not inspect cubes: {e})")
else:
    print("\\nJWST cubes: not downloaded yet")

# 3. Herschel maps
herschel_dir = os.path.join(data_dir, 'herschel')
if os.path.exists(herschel_dir):
    maps = glob.glob(os.path.join(herschel_dir, '**/*.fits'), recursive=True)
    print(f"\\nHerschel maps: {len(maps)} FITS files")
else:
    print("\\nHerschel maps: not downloaded yet")

# 4. Quick-look plot of template spectra
print("\\n--- Quick-Look Plot ---")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fits_files = glob.glob(os.path.join(template_dir, '**/*.fits'), recursive=True)
    if fits_files:
        from astropy.io import fits

        fig, ax = plt.subplots(figsize=(12, 6))

        for fpath in sorted(fits_files)[:5]:
            hdu = fits.open(fpath)
            name = os.path.basename(fpath).replace('.fits', '')
            # Try common column names
            for ext in range(len(hdu)):
                if hdu[ext].data is not None and hasattr(hdu[ext].data, 'dtype'):
                    if hasattr(hdu[ext].data.dtype, 'names') and hdu[ext].data.dtype.names:
                        cols = hdu[ext].data.dtype.names
                        wl_col = [c for c in cols if 'wave' in c.lower() or 'lambda' in c.lower()]
                        fl_col = [c for c in cols if 'flux' in c.lower() or 'data' in c.lower()]
                        if wl_col and fl_col:
                            wl = hdu[ext].data[wl_col[0]]
                            fl = hdu[ext].data[fl_col[0]]
                            ax.plot(wl, fl, label=name, alpha=0.7)
                            break
            hdu.close()

        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel('Flux')
        ax.set_title('PDRs4All Orion Bar — Template Spectra')
        ax.legend(fontsize=8)
        ax.set_yscale('log')

        outpath = os.path.join(data_dir, 'quicklook_templates.png')
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")
    else:
        print("  No template FITS files found for plotting")

except ImportError:
    print("  (Install matplotlib for quick-look plots)")

print("\\n--- PRISM-3D Model Config ---")
config_path = os.path.join(data_dir, 'orion_bar_config.json')
if os.path.exists(config_path):
    import json
    with open(config_path) as f:
        config = json.load(f)
    print(f"  Run command:")
    print(f"    python -m prism3d.run --config {config_path}")
    print(f"  Or with Orion Bar preset:")
    print(f"    python -m prism3d.run --orion-bar --n {config['n']} --output ./orion_run")
else:
    print("  Config not found. Run download_data.py first.")

print("\\nDone.")
'''

    script_path = os.path.join(output_dir, 'inspect_data.py')
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    print(f"  Inspection script: {script_path}")


def check_data(output_dir):
    """Check what data has been downloaded."""
    print("\n" + "="*60)
    print("  Data Inventory Check")
    print("="*60)

    total_size = 0

    for subdir, label in [
        ('templates', 'Template spectra'),
        ('jwst_cubes', 'JWST IFU cubes'),
        ('herschel', 'Herschel maps'),
    ]:
        path = os.path.join(output_dir, subdir)
        if os.path.exists(path):
            files = []
            size = 0
            for root, dirs, fnames in os.walk(path):
                for fn in fnames:
                    fp = os.path.join(root, fn)
                    s = os.path.getsize(fp)
                    files.append(fn)
                    size += s
            print(f"  ✓ {label}: {len(files)} files, {size/1e6:.1f} MB")
            total_size += size
        else:
            print(f"  ✗ {label}: not downloaded")

    for fname in ['orion_bar_config.json', 'orion_bar_parameters.json',
                    'inspect_data.py']:
        if os.path.exists(os.path.join(output_dir, fname)):
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}: missing")

    print(f"\n  Total: {total_size/1e6:.0f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='PRISM-3D Data Downloader — PDRs4All Orion Bar',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start: download template spectra only (~10 MB)
  python download_data.py --templates

  # Full JWST dataset (~5 GB)
  python download_data.py --jwst-cubes

  # Everything
  python download_data.py --all

  # Check what's downloaded
  python download_data.py --check
        """)

    parser.add_argument('--output', type=str, default='./orion_bar_data',
                        help='Output directory (default: ./orion_bar_data)')
    parser.add_argument('--templates', action='store_true',
                        help='Download template spectra only (~10 MB)')
    parser.add_argument('--jwst-cubes', action='store_true',
                        help='Download JWST Level 3 IFU cubes (~5 GB)')
    parser.add_argument('--herschel', action='store_true',
                        help='Download Herschel FIR maps (~500 MB)')
    parser.add_argument('--all', action='store_true',
                        help='Download everything')
    parser.add_argument('--check', action='store_true',
                        help='Check downloaded data inventory')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.check:
        check_data(args.output)
        return

    if not any([args.templates, args.jwst_cubes, args.herschel, args.all]):
        print("No download option specified. Use --help for usage.")
        print("Quick start: python download_data.py --templates")
        return

    print("="*60)
    print("  PRISM-3D Data Downloader")
    print("  Target: Orion Bar (PDRs4All ERS Program 1288)")
    print(f"  Output: {args.output}")
    print("="*60)

    has_all_deps = check_dependencies()

    if args.templates or args.all:
        download_templates(args.output)

    if args.jwst_cubes or args.all:
        download_jwst_cubes(args.output)

    if args.herschel or args.all:
        download_herschel(args.output)

    # Always create config and inspection script
    create_prism3d_config(args.output)
    create_loading_script(args.output)

    print("\n" + "="*60)
    print("  Download Complete")
    print("="*60)
    check_data(args.output)

    print(f"""
Next steps:
  1. Inspect data:   python {args.output}/inspect_data.py
  2. Run PRISM-3D:   python -m prism3d.run --config {args.output}/orion_bar_config.json
  3. Or quick test:  python -m prism3d.run --orion-bar --n 16
""")


if __name__ == '__main__':
    main()
