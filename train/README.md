# Training the Indian plate model

Parda's default plate detector is a YOLO11n fine-tuned on public Indian plate datasets. This folder holds everything needed to rebuild it and to check it.

## Data

We started from eight public datasets on Roboflow Universe, all CC BY 4.0 (listed with credits in [NOTICE.md](../NOTICE.md)). Together they hold 10,936 images, but they overlap heavily: the same photos and video frames are re-uploaded, forked and augmented under different names. A test score on such a mix means little unless the copies are found first.

`prepare_roboflow.py` does that:

1. Keeps only plate boxes, whatever each dataset called its plate class (one names a class per plate number).
2. Groups images that must stay together: copies of the same original file, frames of the same video, photos of the same plate number and near-identical pictures by perceptual hash, mirrored copies included.
3. Keeps one copy of each duplicate. 10,936 images become **3,841 distinct labelled images**.
4. Splits whole groups into train, validation and test, so no copy of a test photo is ever trained on. Images that an uploader squashed to a square go to training only, because they do not look like real camera frames.

| Split | Images | Plates |
|---|---:|---:|
| train | 2,865 | 3,100 |
| validation | 324 | 361 |
| test | 650 | 718 |

The test photos are phone pictures from Indian streets: cars, two-wheelers, autos and trucks, by day and night, with white, yellow, green and black plates.

### Cleaning the labels

Some images were stored sideways while their boxes were drawn on the upright picture, so the box lands on empty road. `fix_rotation.py` finds images where a stock plate detector matches none of the boxes, tries the other three rotations and keeps the one where detector and labels agree. It straightened 86 test images. We then looked at every one of the 35 test images the detector matched in no rotation. 27 are real plates the detector simply misses, and they stay as they are. 6 were sideways with plates too small to confirm automatically, so we rotated them by hand. 2 had boxes that sit on no plate, so we removed them. The decisions are in [review/test_review.json](review/test_review.json) and `apply_review.py` applies them.

Using the stock detector to choose rotations can only help the stock detector's score, so it does not flatter the new model.

## Training

```bash
pip install ultralytics imagehash
python train/prepare_roboflow.py --src downloads/ --out data/plates
python train/fix_rotation.py data/plates
python train/apply_review.py data/plates train/review/test_review.json
python train/finetune_plates.py --data data/plates/data.yaml --imgsz 384 --epochs 40
```

YOLO11n from COCO weights, 40 epochs at 384 pixels, the Ultralytics defaults plus stronger scale and brightness jitter. It trained on a two-core CPU in about 6.5 hours. The best checkpoint (epoch 35) scores mAP50 0.92 on the validation split.

## Choosing the threshold

A detector's score threshold trades missed plates against extra masking. We chose it on the validation split, not the test split: the lowest threshold at which the new model masks no more of the picture outside plates than the stock model does. That is 0.05 for the whole-frame pass. Inside the native-scale tiles of `--thorough` mode the model sees many more small regions, so there it uses 0.5, again chosen on validation.

## Results on the held-out test split

650 images and 718 plates the model never saw, scored by Parda's own metric: a plate counts as protected when at least 80% (or 95%) of its area is masked. Over-mask is the share of the picture masked outside any labelled plate.

| Plate model | Protected at 80% | Protected at 95% | Large plates | Small plates | Tiny plates | Over-mask | Time per frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| open-image-models YOLOv9-t 640 (previous Python default) | 78.8% | 71.7% | 68.9% | 87.6% | 96.4% | 1.19% | 100 ms |
| open-image-models YOLOv9-t 384 (previous browser default) | 81.5% | 73.3% | 72.7% | 90.0% | 92.9% | 1.53% | 46 ms |
| **Indian YOLO11n 384 (new default)** | **87.9%** | **73.8%** | **81.8%** | **94.3%** | 92.9% | 1.53% | 40 ms |

Large plates are 363 of the test plates, small 299 and tiny 56 (under 16 pixels tall). Times are for whole-frame passes on frames up to 1920 pixels, on a two-core cloud CPU.

What this says:

* At the same over-mask, the fine-tuned model protects 6.4 more plates in every 100 than the model it replaces. It is also a little faster.
* The gain is largest on large plates, which in this test set are mostly two-wheeler, auto and EV plates photographed up close. Those were the stock model's weak spot.
* At 95% coverage the two are level: the new model's boxes are tighter, so fully covering a plate depends more on the mask margin.
* One source of test photos is much harder than the rest. On its 278 images and 344 plates, the new model protects 77.9% against the stock model's 63.4%. Many of those are two-wheeler plates at steep angles and photos taken sideways.
* On the easier source (363 images, 365 plates) the new model is a point behind: 97.0% against 98.1%. Fine-tuning did not win everywhere.

### Thorough mode

`--thorough` adds native-scale tiles to the whole-frame pass. The browser demo always uses it for photos.

| Plate model, thorough | Protected at 80% | Protected at 95% | Over-mask | Time per frame |
|---|---:|---:|---:|---:|
| YOLOv9-t 640 (previous Python) | 82.7% | 73.1% | 1.29% | 516 ms |
| YOLOv9-t 384 (previous browser photos) | 85.0% | 75.8% | 18.5% | 468 ms |
| **Indian YOLO11n 384 (new)** | **88.6%** | 74.8% | 1.78% | 341 ms |

The old browser setting masked almost a fifth of every photo outside the plates: inside tiles, the stock 384 model fired on signs, text and texture. The new setting cuts that tenfold and protects more plates. With the new model the tiles add little, 0.7 points for eight times the time, so `--thorough` is now mainly for distant plates in wide dashcam frames.

## Limits

* These datasets are mostly phone photos of parked vehicles, not dashcam video at speed. A dashcam test set is the next step ([data/README.md](../data/README.md)).
* Labels come from many uploaders and are not perfect. Images with a plate left unlabelled count every mask on that plate as over-mask, so over-mask here is an upper bound.
* The model is trained with Ultralytics and is therefore AGPL-3.0 (see [NOTICE.md](../NOTICE.md)).

## Files

```
prepare_roboflow.py   merge, deduplicate and split Roboflow YOLO exports
fix_rotation.py       straighten images stored sideways relative to their labels
apply_review.py       apply the manual test review in review/
finetune_plates.py    train with Ultralytics and export ONNX for Parda
eval_plates.py        score plate models on a test split with Parda's metrics
results/              the numbers behind the tables above
```
