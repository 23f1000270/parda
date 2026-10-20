# Third-party models and data

Parda's own code is licensed under Apache-2.0 (see LICENSE). It uses these models, which keep their own licences:

| File | Model | Source | Licence |
|---|---|---|---|
| `site/models/face-yunet-640.onnx` | YuNet face detector (2023mar) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), Shiqi Yu et al. | MIT |
| `site/models/plate-indian-yolo11n-384.onnx` | YOLO11n plate detector fine-tuned on Indian plates (default) | Trained in this repo with [Ultralytics](https://github.com/ultralytics/ultralytics) from its COCO-pretrained YOLO11n, on the datasets below | AGPL-3.0 |
| `site/models/plate-yolov9t-384.onnx` | YOLOv9-t licence plate detector, end-to-end | [open-image-models](https://github.com/ankandrew/open-image-models), ankandrew | MIT |

The Python package downloads the same models on first use (plus the 640-pixel open-image-models plate model) and checks each file against a pinned SHA-256 before loading it.

**About the AGPL-3.0 model.** Ultralytics licenses its code and the weights it produces under AGPL-3.0, so the fine-tuned plate model is AGPL-3.0 while the rest of Parda stays Apache-2.0. The model is a separate file. If that licence does not suit your use, run Parda with `--plate-model plate-yolov9t-640` (MIT) or train your own model with a framework whose licence you prefer.

## Training data for the Indian plate model

The fine-tuned plate model was trained and tested on these public datasets from Roboflow Universe, all published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Licences are as stated by each uploader. We do not redistribute any of the images.

| Dataset | Author on Roboflow Universe | Version used |
|---|---|---|
| [Indian number plate](https://universe.roboflow.com/quobotic/indian-number-plate) | quobotic | 3 |
| [Indian Number Plate Detection](https://universe.roboflow.com/bharani-ejlrn/indian-number-plate-detection) | Bharani | 3 |
| [Indian Number Plate Detection](https://universe.roboflow.com/payals-workspace-gz1ub/indian-number-plate-detection-g7voe) | Payal's Workspace | 1 |
| [indian-number-plate-v1](https://universe.roboflow.com/aihelmet-brl8g/indian-number-plate-v1) | aihelmet | 3 |
| [Indian Number Plate](https://universe.roboflow.com/ayan-tvtud/indian-number-plate-vpgcg) | Ayan | 2 |
| [Indian number plate](https://universe.roboflow.com/i5hub/indian-number-plate-keeo5-zllyl) | i5HUB | 1 |
| [Indian Cars Number Plate](https://universe.roboflow.com/ananth11/indian-cars-number-plate) | Ananth11 | 1 |
| [indian automatic number plate](https://universe.roboflow.com/coep-4emgc/indian-automatic-number-plate) | coep | 2 |

We thank every author. How we merged, cleaned and split them is described in [train/README.md](train/README.md).
