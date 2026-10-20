# The Indian roads test set

Parda is only as good as its worst miss. To measure that honestly we are building a small, carefully labelled test set from Indian roads, scored with `parda bench`.

## What goes in

Frames are sampled from dashcam and phone video, at most one frame every two seconds so the set is not dominated by near-duplicates. We aim for a spread across:

| Condition | Values |
|---|---|
| `light` (per image) | day, dusk, night |
| `weather` (per image) | clear, rain, haze |
| `scene` (per image) | highway, city, market, campus |

Every face and every number plate in a frame is boxed, including tiny, blurred and partly hidden ones. Each box carries attributes:

| Label | Attribute | Values |
|---|---|---|
| plate | `plate_type` | one_line, two_line, hand_painted, other |
| plate | `colour` | white, yellow, green, black, other |
| plate | `vehicle` | car, two_wheeler, auto, bus_truck, other |
| face | `view` | frontal, profile, back, helmet |
| both | `size` | tiny (under 16 px), small (16 to 48 px), large |

Results are reported per attribute, so a weakness on two-line plates or helmeted riders cannot hide behind a good overall number.

## Privacy of the test set itself

A test set of unblurred faces and plates is exactly what Parda exists to prevent. So:

* Only people and vehicles who have agreed to take part appear in any frame we publish, or the frame stays private and only its labels and our scores are published.
* We never publish frames from public roads that show bystanders.
* Anyone can ask for their frames to be removed.

## Labelling

We label in [CVAT](https://github.com/cvat-ai/cvat) with two box labels, `face` and `plate`, and the attributes above. Export as COCO 1.0, then:

```bash
parda convert-cvat annotations/instances_default.json my_set/labels.json
parda bench my_set --json results.json
```

## Scoring other tools

Export the regions another tool masked as `{"frames/0001.jpg": [[x1, y1, x2, y2], ...]}` and run `parda bench my_set --predictions theirs.json`. The same metric then applies to every tool.

## Status

Collection has started. Until it is done, plate results on a held-out split of public Indian plate datasets are in [train/README.md](../train/README.md). That split is phone photos of mostly parked vehicles and has no face labels, so it does not replace this test set.
