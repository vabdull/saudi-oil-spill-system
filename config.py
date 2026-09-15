"""Central configuration for data preparation, inference, and presentation.

The production path imports this module directly. Keeping operational values in
one place makes configuration drift visible and allows the backend cache to hash
the settings that can alter an analysis. Operating values reproduce the published
CUDA-bf16 results and must not be tuned on the demonstration scenes. Changing one
requires deliberate review and an explicit update to the reference checksums.
"""

from pathlib import Path

# Absolute repository root derived from this file; valid on any local checkout.
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing single-band demo GeoTIFFs; must remain below PROJECT_ROOT.
DEMO_SCENES_DIR = PROJECT_ROOT / "data" / "demo_scenes"
# Natural Earth 1:10m coastline path; the .dbf/.shx/.prj/.cpg siblings are required.
COASTLINE_PATH = PROJECT_ROOT / "data" / "natural_earth" / "ne_10m_coastline.shp"
# Initial offshore coastline search margin in degrees (>0); 2° covers the roughly
# 95 km Jeddah offset, with a single 5° fallback implemented in src.measure.
COASTLINE_SEARCH_BUFFER_DEG = 2.0

# Square model input edge in pixels (>0); fixed by the trained checkpoints at 256.
TILE_SIZE = 256
# Sliding-window stride in pixels (0 < STRIDE <= TILE_SIZE); 224 leaves 32 px overlap.
STRIDE = 224
# Oil-class probability threshold in [0,1]; selected at the validated epoch-17 point.
SCREENER_THRESHOLD = 0.45

# Normalization strategy: one of "scene", "tile", or "fixed_db". Scene scope is
# used because per-tile stretching makes uniformly dark slick interiors resemble
# ordinary water and materially inflated the synthetic-scene result.
NORM_SCOPE = "scene"
# Lower robust scene percentile in [0,100); set to 2% by the training pipeline.
NORM_LO_PCT = 2.0
# Upper robust scene percentile in (NORM_LO_PCT,100]; set to 98%.
NORM_HI_PCT = 98.0
# Maximum deterministic percentile sample count (>0); bounds RAM and runtime.
NORM_MAX_SAMPLES = 2_000_000
# Lower calibrated-backscatter bound in dB (< FIXED_DB_HI); diagnostic fixed-dB mode.
FIXED_DB_LO = -30.0
# Upper calibrated-backscatter bound in dB (> FIXED_DB_LO); diagnostic fixed-dB mode.
FIXED_DB_HI = 0.0

# Normalized-intensity cutoff in [0,1]; selected at 0.30 from validation and used by
# the classical union only inside the final screener footprint.
BASELINE_THRESHOLD = 0.30
# Learned segmenter probability cutoff in [0,1]; selected on validation at 0.10.
MASK_THRESHOLD = 0.10
# Binary-opening disk radius in pixels (integer >=0); configured at 2 pixels.
MORPH_OPENING_RADIUS = 2
# Minimum retained connected component in pixels (integer >=1); 1000 px is 0.1 km²
# on the 10 m demo grids and is the released post-processing value.
MIN_BLOB_PX = 1_000
# skimage connectivity (1 or 2 for 2D); 2 preserves diagonally connected oil pixels.
BLOB_CONNECTIVITY = 2
# Select the trained two-stage path; False is reserved for deterministic test baselines.
USE_TRAINED_OPERATING_MODELS = True
# Inference batch size in tiles (integer >=1); 64 fits the reference 16 GB GPU.
OPERATING_BATCH_SIZE = 64
# Require CUDA bf16 for published numerical equivalence; CPU fp32 is diagnostic only.
OPERATING_REQUIRE_CUDA_BF16 = True
# Trained epoch-17 ResNet-18 screener checkpoint from saudi-oil-spill-assets.zip.
OPERATING_SCREENER_CHECKPOINT = PROJECT_ROOT / "models" / "screener_epoch_17.pt"
# Trained epoch-14 ResNet-34 U-Net checkpoint from saudi-oil-spill-assets.zip.
OPERATING_SEGMENTER_CHECKPOINT = PROJECT_ROOT / "models" / "segmenter_selected.pt"
# Apply 3x3 binary closing to the tile pass map; enabled after paired-gate evaluation.
SCREENER_PASS_MAP_CLOSING = True
# Fill enclosed holes in the tile pass map; enabled to recover boundary-enclosed tiles.
SCREENER_PASS_MAP_HOLE_FILLING = True
# Union the learned mask with the validated dark-pixel baseline inside passed tiles.
CLASSICAL_UNION_ENABLED = True
# Never allow the classical baseline outside Stage 1's final pass footprint; this
# containment is what prevents the baseline from freely accepting look-alikes.
CLASSICAL_UNION_RESTRICT_TO_PASS_FOOTPRINT = True
# Do not fill holes in the final pixel mask; validation rejected that alternative.
CHAIN_BINARY_FILL_HOLES = False

# Oil-overlay opacity in [0,1]; presentation-only value selected for SAR readability.
OVERLAY_ALPHA = 0.42
# Rejected-tile overlay opacity in [0,1]; presentation-only yellow tint.
REJECTED_OVERLAY_ALPHA = 0.24
# Global pseudorandom seed (non-negative integer) used for reproducible data/training.
RANDOM_SEED = 42
# Maximum invalid fraction in [0,1]; training and inference both reject only >20%,
# so a tile containing exactly 20% invalid pixels remains eligible.
MAX_INVALID_FRACTION = 0.20
# Replace invalid pixels in eligible tiles with the valid scene median before
# normalization; this prevents nodata from clipping to maximally dark apparent oil.
INFILL_INVALID_PIXELS = True
# Diagnostic boundary-ring exclusion switch. Set False because the ring removed
# 10.74% of Jeddah; median infill corrects nodata without discarding that area.
EXCLUDE_BOUNDARY_ADJACENT_TILES = False
# Broad display regions in longitude/latitude degrees; each interval is ordered
# (minimum, maximum) and limited to valid WGS84 ranges.
REGION_BOUNDS = {
    "Arabian Gulf": {"lon": (47.0, 57.0), "lat": (22.0, 31.0)},
    "Red Sea": {"lon": (32.0, 44.0), "lat": (12.0, 30.0)},
}
# Validated focus bounds in longitude/latitude degrees, derived from the packaged
# Saudi demo footprints and nested inside the corresponding broad region.
VALIDATED_REGION_BOUNDS = {
    "Arabian Gulf": {"lon": (48.0, 55.0), "lat": (24.0, 30.0)},
    "Red Sea": {"lon": (37.0, 42.0), "lat": (18.0, 28.0)},
}

# Ordered impact labels from least to most severe; used only for scene aggregation.
SEVERITY_ORDER = ("Low", "Moderate", "High", "Critical")
# Medium learned-support cutoff in [0,1]; midpoint of the observed 0.229–0.292 gap.
CONFIDENCE_MEDIUM_THRESHOLD = 0.26
# High learned-support cutoff in [0,1]; midpoint of the observed 0.674–0.729 gap.
CONFIDENCE_HIGH_THRESHOLD = 0.70
# Ordered learned-support labels from least to most supported; not calibrated odds.
CONFIDENCE_ORDER = ("Low", "Medium", "High")
