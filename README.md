# Saudi Oil Spill Detection System

Two-stage detection and measurement of oil spills in Sentinel-1 SAR imagery,
with a web dashboard for analyst review.

A ResNet-18 screener classifies each 256×256 tile as oil, look-alike or no-oil.
A ResNet-34 U-Net then segments the surviving tiles at pixel level, combined
with a dark-intensity rule applied only inside the screener footprint. Each
detected region is measured for area, position, distance to coast and alert
level.

---

## Install and run

**Requirements:** Windows, Python 3.11+, NVIDIA GPU with CUDA bf16. CPU
execution works but is not numerically equivalent on threshold-sensitive
scenes.

```powershell
git clone https://github.com/vabdull/saudi-oil-spill-system
cd saudi-oil-spill-system
```

Download [`saudi-oil-spill-assets.zip`](https://github.com/vabdull/saudi-oil-spill-system/releases/latest)
(1.28 GiB) into the repository root and extract it:

```powershell
Expand-Archive -Path saudi-oil-spill-assets.zip -DestinationPath . -Force
```

This places the model checkpoints in `models/`, the four demo scenes in
`data/demo_scenes/`, and eight upload scenes with reference masks in
`test_scenes/`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-frontend.txt
python -m backend.main
```

Open <http://127.0.0.1:8000>. A scene processes in 1.6–3.3 s on an
RTX 4070 Ti Super.

---

## Results

Trujillo Part III, audited before use: 245 of 450 scenes removed for geographic
overlap with training data (214), near-duplication (7) or data quality (24).
The surviving 205 scenes are 47 oil, 93 look-alike, 65 no-oil.

| Measure | Result |
| --- | ---: |
| Chained oil-scene IoU (47 scenes) | **0.6567** |
| Chained precision | 0.8319 |
| Chained recall | 0.7572 |
| Look-alike tile rejection (6,691 / 7,330) | **91.28%** |
| Look-alike scene rejection (33 / 93) | 35.48% |
| Mean look-alike false-alarm area | 6.65 km² |
| No-oil scenes with zero false alarm | **59 / 65** |
| Segmenter tile-level IoU | 0.7440 |

Tile rejection measures how much of a look-alike scene the screener removes;
scene rejection measures how often a whole scene stays quiet, and is stricter
because one surviving tile flags the scene.

This is an in-family test set, not a regional one. No labelled Saudi-region
test set exists.

### Intensity component, with and without

| Measure | With | Without |
| --- | ---: | ---: |
| Oil-scene IoU | 0.6567 | 0.6709 |
| Precision | 0.8319 | 0.8720 |
| Recall | 0.7572 | 0.7442 |
| Look-alike false-alarm area | 618.88 km² | 324.79 km² |
| No-oil false-alarm area | 5.17 km² | 3.39 km² |

IoU difference −0.0142, paired bootstrap interval [−0.052, +0.012], which spans
zero. The effect is shape-dependent: Spearman ρ = +0.798 against truth oil
fraction, −0.785 against perimeter-to-area ratio. Part III is dominated by thin
slicks.

### Demo scenes

| Scene | Stage 1 footprint km² | Chained km² | Regions |
| --- | ---: | ---: | ---: |
| Jeddah, 13 Oct 2019 (SABITI) | 427.9978 | 161.2996 | 7 |
| Jeddah, 25 Oct 2019 (no oil) | 592.1821 | 17.7830 | 36 |
| Arabian Gulf control (no oil) | 54.0347 | 6.9909 | 3 |
| Kuwait, Aug 2017 (Al-Khafji) | 281.1591 | 31.4344 | 18 |

On the 25 October scene, presumed to contain no oil, Stage 1 passes a footprint
covering 592 km² and the chain reports 17.78 km², a 97% reduction. The two oil
scenes are qualitative: no independent reference mask exists for either.

---

## Operating configuration

| Component | Value |
| --- | --- |
| Screener | ResNet-18, epoch 17, P(oil) ≥ 0.45 |
| Invalid-pixel rule | reject tile above 20% invalid; scene-median infill otherwise |
| Pass-map morphology | 3×3 closing, then hole filling |
| Segmenter | U-Net / ResNet-34, epoch 14, pixel threshold 0.10 |
| Intensity union | normalised intensity < 0.30, inside pass footprint only |
| Post-processing | opening radius 2, remove components below 1,000 px |
| Tiling | 256×256, stride 224 |
| Normalisation | scene-scope 2nd–98th percentile over valid pixels |
| Runtime | CUDA bf16, batch size 64, seed 42 |

---

## Data sources

| Dataset | Record | Role |
| --- | --- | --- |
| Refined Deep-SAR Oil Spill (SOS) | [Zenodo 15298010](https://zenodo.org/records/15298010) | Persian Gulf Sentinel-1 oil and no-oil training tiles |
| Trujillo Part I | [Zenodo 8346860](https://zenodo.org/records/8346860) | 1,200 calibrated-dB oil scenes with masks |
| Trujillo Part II | [Zenodo 8253899](https://zenodo.org/records/8253899) | 685 look-alike and 685 no-oil scenes |
| Trujillo Part III | [Zenodo 13761290](https://zenodo.org/records/13761290) | Held-out test set with author-drawn masks |
| Natural Earth coastline | [naturalearthdata.com](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/) | Distance-to-coast measurement |

Oil training tiles are drawn 1,330 from each source and matched bin-by-bin on
oil fraction, so that neither dataset origin nor slick size predicts the class.

**Demo scenes** were exported from Google Earth Engine (`COPERNICUS/S1_GRD`):
IW mode, VV polarisation, descending pass, 10 m resolution, UTM projection set
explicitly, 25 m focal median applied in linear power.

| Scene | Location | Acquired |
| --- | --- | --- |
| `jeddah_oct_2019` | Red Sea off Jeddah | 13 October 2019 |
| `jeddah_oct25_2019` | Same footprint | 25 October 2019 |
| `gulf_open_water_oct2019` | Arabian Gulf off Jubail | October 2019 |
| `kuwait_aug_2017` | Arabian Gulf off Kuwait | August 2017 |

---

## Dataset citations

- Q. Zhu, Y. Zhang, Z. Li, X. Yan, Q. Guan, Y. Zhong, L. Zhang, and D. Li,
  "Oil spill contextual and boundary-supervised detection network based on
  marine SAR images," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, pp. 1-10,
  2021. [doi:10.1109/TGRS.2021.3115492](https://doi.org/10.1109/TGRS.2021.3115492)
- R. Trujillo-Acatitla, J. Tuxpan-Vargas, C. Ovando-Vazquez, and
  E. Monterrubio-Martinez, "Marine oil spill detection and segmentation in SAR
  data with two steps deep learning framework," *Mar. Pollut. Bull.*, vol. 204,
  p. 116549, 2024. [doi:10.1016/j.marpolbul.2024.116549](https://doi.org/10.1016/j.marpolbul.2024.116549)

---

## Author

**Abdullah Sultan Alotaibi**, Cooperative Training Programme, Space Technology
Institute, King Abdulaziz City for Science and Technology (KACST).
Supervised by **Mr. Ibrahim Alrayes**.
