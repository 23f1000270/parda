# Third-party models

Parda's own code is licensed under Apache-2.0 (see LICENSE). It uses these pretrained models, which keep their own licences:

| File | Model | Source | Licence |
|---|---|---|---|
| `site/models/face-yunet-640.onnx` | YuNet face detector (2023mar) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), Shiqi Yu et al. | MIT |
| `site/models/plate-yolov9t-384.onnx` | YOLOv9-t licence plate detector, end-to-end | [open-image-models](https://github.com/ankandrew/open-image-models), ankandrew | MIT |

The Python package downloads the same models on first use (plus the 640-pixel plate model) and checks each file against a pinned SHA-256 before loading it.
