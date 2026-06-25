import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

import rasterio
import numpy as np
import matplotlib
# Use 'Agg' backend for headless/server rendering; comment out for interactive Jupyter/Colab sessions.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# Global map rendering configuration
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 14, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 14, 'axes.labelweight': 'bold', 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.linewidth': 1.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42
})


def plot_global_tiff(tiff_path, out_pdf_name, title, cmap, vmin, vmax, cbar_label):
    """Render a global GeoTIFF on a Cartopy map and export as PDF."""
    print(f"Rendering {title} ...")

    # 1. Read TIFF data and geographic extent
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        # Extract extent [left, right, bottom, top] for Cartopy alignment
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Mask NaN values for transparency, allowing underlying land/ocean colours to show through
    data_masked = np.ma.masked_invalid(data)

    # 2. Create Cartopy map canvas
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Restrict view to the actual export extent (exclude Antarctic blank margin)
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())

    # 3. Add geographic features (aligned with Fig 1a)
    ax.add_feature(cfeature.LAND, facecolor='#f4f4f4', alpha=0.9, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0e6ed', alpha=0.5, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.4, zorder=3)  # Coastline on top (zorder=3) to overlay pixels cleanly

    # 4. Render TIFF pixels onto the map
    # transform=ccrs.PlateCarree() is critical: it tells Cartopy that the layer uses geographic coordinates.
    im = ax.imshow(data_masked, origin='upper', extent=extent,
                   transform=ccrs.PlateCarree(), cmap=cmap,
                   vmin=vmin, vmax=vmax, zorder=1, interpolation='none')

    # 5. Add horizontal colour bar at the base of the figure
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        fraction=0.03, pad=0.06, aspect=40)
    cbar.set_label(cbar_label, fontweight='bold')

    # Export as high-resolution PDF
    plt.tight_layout()
    plt.savefig(out_pdf_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf_name}\n")


# ==========================================
# Generate figures
# ==========================================

# Figure 4a: Empty Shell syndrome probability map
plot_global_tiff(
    tiff_path=os.path.join(TEMP_DIR, "Fig4a_Global_ES_Probability.tif"),
    out_pdf_name=os.path.join(TEMP_DIR, "Fig4a_Map_Cartopy.pdf"),
    title="Fig 4a. Global Potential Risk of Empty Shell Syndrome",
    cmap='inferno',  # Alternative: 'Reds'
    vmin=0,
    vmax=1,
    cbar_label='Probability of ES Syndrome'
)

# Figure 4b: Phantom carbon deficit map (units: Mg C per 5 km pixel)
plot_global_tiff(
    tiff_path=os.path.join(TEMP_DIR, "Fig4b_Global_Phantom_AGB_Deficit.tif"),
    out_pdf_name=os.path.join(TEMP_DIR, "Fig4b_Map_Cartopy.pdf"),
    title="Fig 4b. Global Phantom Carbon Sink Deficit",
    cmap='YlOrRd',
    vmin=0,
    vmax=5500,  # 99th percentile, clips extreme tail
    cbar_label='Phantom Carbon Deficit (Mg C / 5km pixel)'
)

print("All maps generated.")
