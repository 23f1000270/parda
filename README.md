# Parda पर्दा

Parda hides faces and number plates in photos and video before anyone else sees them. It runs on your own device: in a browser tab, on a laptop, or on a Raspberry Pi 5 dashcam that masks every frame before it is saved.

**Try it:** [23f1000270.github.io/parda](https://23f1000270.github.io/parda/) (camera, photo or video; nothing is uploaded)

## Why we built it

Road video is the raw material for road-safety research and driver-assistance systems. In India it is also full of personal data. Under the Digital Personal Data Protection Act, 2023, faces and number plates identify people, which makes sharing Indian road footage risky for dashcam owners, researchers and startups. Less shared footage means less open data, and AI for Indian roads falls further behind.

Good open tools for this already exist, such as Meta's [EgoBlur](https://github.com/facebookresearch/EgoBlur) and [dashcamcleaner](https://github.com/tfaehse/dashcamcleaner). They were built and tested mostly on Western roads. Indian roads add two-line plates, plates on two-wheelers and autos, yellow commercial and green EV plates, BH-series and hand-painted plates, helmeted riders and dense crowds at night. Parda is built and measured for those conditions.

## What it does

* **Finds faces and plates on-device.** YuNet for faces and a YOLOv9-t model for plates, both running on ONNX Runtime. Nothing leaves the machine.
* **Scans twice.** One pass over the whole frame for large objects, then full-resolution tiles so small faces in a crowd and distant plates are not shrunk to nothing. Photos in the browser get both passes; the CLI can do both on every video frame.
* **Masks so it cannot be undone.** Blur and fine pixelation can be partly reversed by learned models ([McPherson et al., 2016](https://arxiv.org/abs/1609.00408)). Parda replaces each region with at most six cells of averaged colour plus random noise.
* **Remembers what flickers.** Detectors miss objects for a frame or two. Parda keeps masking a lost face or plate for eight more frames, moves the mask along its last motion and grows it each frame, so uncertainty always gets a bigger mask, never a smaller one.
* **Fails closed.** If a model crashes on a frame, that whole frame is hidden.
* **Drops what else identifies you.** Saved videos have no audio track and no metadata, so voices and the GPS tracks many dashcams embed are gone too. Saved photos carry no EXIF.
* **Masks at capture.** In live mode, frames are masked in memory and only then written, so the raw video never exists on disk. If the device falls behind, it drops frames rather than saving them unmasked. Live mode is tested with webcams and video streams; Raspberry Pi 5 camera tests are under way.

```mermaid
flowchart LR
  A[camera or file] --> B[frame in memory]
  B --> C[whole-frame pass]
  B --> D[full-resolution tiles]
  C --> E[merge and NMS]
  D --> E
  E --> F[mask memory: hold, follow, grow]
  F --> G[coarse noisy mask]
  G --> H[write frame, no audio, no metadata]
  C -. model error .-> X[hide whole frame]
```

## Use it

```bash
pip install git+https://github.com/23f1000270/parda
parda fetch-models                     # downloads and checksums the models

parda image photos/ -o blurred/        # a folder of photos
parda video drive.mp4 -o drive_safe.mp4
parda video drive.mp4 -o drive_safe.mp4 --thorough   # also tile-scan for distant plates
parda live --source picamera2 --out-dir rec/          # Raspberry Pi 5 camera
parda live --source 0 --preview --seconds 30          # any webcam
```

From Python:

```python
import cv2
from parda import Anonymizer

anon = Anonymizer()
result = anon.process(cv2.imread("street.jpg"), use_memory=False)
cv2.imwrite("street_safe.jpg", result.image)
```

Every model is fetched from its upstream release and checked against a pinned SHA-256 before it is loaded. You can also pass your own model with `--plate-model my_model.onnx`; `train/finetune_plates.py` fine-tunes one on Indian plates and exports it in a format Parda reads directly.

## How we measure it

Detection benchmarks reward tight boxes. Privacy asks a different question: after masking, how much of each face or plate can still be seen? So `parda bench` reports:

* **Protected recall at t:** the share of labelled faces and plates with at least a fraction t of their area masked (we report t = 0.80 and 0.95).
* **Over-mask:** how much of the frame was masked outside any labelled object, which is the picture quality we gave up.

Both are broken down by every label attribute: plate type and colour, vehicle, face view, object size, light and weather. Any other tool can be scored the same way from a JSON of the regions it masked. See [data/README.md](data/README.md) for the test set protocol.

**Results:** our Indian roads test set is being collected and labelled now. We will publish numbers for Parda and EgoBlur here once it is done, including the cases where Parda loses.

## Known limits

* YuNet is weak on faces in profile, from behind and under helmets. A person-detector fallback that masks the head region is next on the list.
* The plate model sometimes masks text that looks like a plate. That costs a little picture, not privacy.
* The browser video mode masks one fast pass per frame and records in real time, so slow devices drop frames. Use the CLI for full-quality output.
* Parda reduces risk; it is not a legal compliance guarantee. Check what your use of the footage requires.

## Roadmap

1. Indian roads test set and first public results.
2. Plate model fine-tuned on Indian plates.
3. Head fallback for helmeted riders and back-of-head views.
4. Measured frame rates on Raspberry Pi 5 and NVIDIA Jetson.

## Project layout

```
parda/        Python package: detectors, mask memory, masks, video, live capture, benchmark
site/         the in-browser demo (ONNX Runtime Web, no build step)
train/        fine-tuning script for an Indian plate detector
data/         test set protocol and labelling guide
tests/        pytest suite
```

## Licence

Apache-2.0 for Parda's code. The bundled models are MIT; see [NOTICE.md](NOTICE.md).
