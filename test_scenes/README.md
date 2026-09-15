# Part III manual-upload test scenes

These eight oil scenes come only from the accepted 47-scene surviving Part III subset. They span the system's recorded performance range and exclude every scene removed by the geographic-overlap, near-duplicate, band-quality, or exact-zero audit.

Each `*_vv.tif` is a single-band upload image. VV was selected independently per scene as the brighter of the two source bands over finite pixels, excluding locations where both bands are exactly zero; those both-zero locations are declared nodata (`0.0`) in the export. Each `*_truth.tif` contains the published author mask copied onto exactly the corresponding image CRS, transform, dimensions, and pixel grid.

## Scene index

| VV file | Truth file | Truth area (km²) | System area (km²) | IoU | Precision | Recall | What it demonstrates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `00111_vv.tif` | `00111_truth.tif` | 68.0548 | 67.2275 | 0.9539 | 0.9824 | 0.9705 | Best case: near-complete, precise recovery at the top of the held-out IoU range. |
| `00097_vv.tif` | `00097_truth.tif` | 21.5169 | 20.8178 | 0.8686 | 0.9453 | 0.9146 | High/mid-range case whose predicted and published truth areas closely agree. |
| `00063_vv.tif` | `00063_truth.tif` | 4.3628 | 5.8456 | 0.7408 | 0.7431 | 0.9958 | Upper-middle case near 0.74 IoU, filling the gap below 00097. |
| `00083_vv.tif` | `00083_truth.tif` | 27.1180 | 24.7424 | 0.6328 | 0.8123 | 0.7412 | Middle case near 0.63 IoU with both false positives and missed truth. |
| `00078_vv.tif` | `00078_truth.tif` | 29.2799 | 16.8213 | 0.4967 | 0.9095 | 0.5225 | Lower-middle case near 0.50 IoU, dominated by under-detection. |
| `00054_vv.tif` | `00054_truth.tif` | 102.3319 | 42.7284 | 0.3988 | 0.9681 | 0.4042 | Required large-slick under-detection case with recall near 0.404. |
| `00074_vv.tif` | `00074_truth.tif` | 9.8031 | 27.0452 | 0.2827 | 0.3003 | 0.8283 | Low-IoU over-detection case filling the range above the worst scene. |
| `00037_vv.tif` | `00037_truth.tif` | 2.4542 | 28.1415 | 0.0601 | 0.0616 | 0.7064 | Worst surviving case: heavy over-detection and approximately 0.06 IoU. |

## Empirical band selection

| Scene | Band 1 mean (dB) | Band 2 mean (dB) | Selected VV band | Separation (dB) | Below 1.0 dB? |
| --- | ---: | ---: | ---: | ---: | --- |
| 00111 | -32.6566 | -24.9636 | 2 | 7.6930 | No |
| 00097 | -32.1733 | -23.3199 | 2 | 8.8534 | No |
| 00063 | -31.6282 | -22.0567 | 2 | 9.5715 | No |
| 00083 | -28.3663 | -23.5033 | 2 | 4.8631 | No |
| 00078 | -28.2307 | -18.4340 | 2 | 9.7967 | No |
| 00054 | -30.9952 | -21.2144 | 2 | 9.7807 | No |
| 00074 | -26.5300 | -16.6192 | 2 | 9.9108 | No |
| 00037 | -30.5703 | -21.6120 | 2 | 8.9583 | No |

## Production upload verification

The files were submitted through `POST /api/analyze`, the same FastAPI entry point used by the browser upload control. `App mask area` below remeasures the returned production mask geodesically on its preserved EPSG:4326 grid; `dashboard area` is the API's current `total_area_km2` field.

| Scene | Evaluation area (km²) | App mask area, geodesic (km²) | Mask-area match | Dashboard area (km²) | Dashboard-area match |
| --- | ---: | ---: | --- | ---: | --- |
| 00111 | 67.227483 | 67.227483 | PASS | 67.227483390825 | PASS |
| 00097 | 20.817794 | 20.817794 | PASS | 20.817794085957 | PASS |
| 00063 | 5.845644 | 5.845644 | PASS | 5.845644279030 | PASS |
| 00083 | 24.742394 | 24.742394 | PASS | 24.742394237120 | PASS |
| 00078 | 16.821346 | 16.821346 | PASS | 16.821346192112 | PASS |
| 00054 | 42.728367 | 42.728367 | PASS | 42.728366824202 | PASS |
| 00074 | 27.045152 | 27.045152 | PASS | 27.045152010539 | PASS |
| 00037 | 28.141452 | 28.141452 | PASS | 28.141452275948 | PASS |

Folder size: 97,955,748 bytes (93.42 MiB).
