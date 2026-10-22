# Computer Vision Robot

Real-time traffic-sign recognition for a robot/controller. The system captures
camera frames, recognizes traffic signs with an Ultralytics YOLO model, and
prints a movement command after a stable prediction. It can be demonstrated
with a webcam or a phone stream even when the robot is not connected.

## Project information

| Item | Details |
|---|---|
| Project specialization | Traffic Sign Recognition |
| Team | Albano, Luis Van Dirk; Borac, John Joshua; Carullo, John Wilbert; Regalado, Roger Jr.; Reyes, Reynaldo Jr.; Sabio, Jacob Angel |
| Final inference entry point | [`src/testing/testing.py`](src/testing/testing.py) |
| Default weights | [`models/best.pt`](models/best.pt) |

## System overview

The implementation follows **SEE -> DECIDE -> ACT**:

1. **SEE:** Open a webcam or phone stream, rotate the frame when necessary,
   resize very large frames, and display the live image.
2. **DECIDE:** Run YOLO detection or classification, discard predictions below
   the `0.80` confidence threshold, suppress overlapping scan windows, and
   choose the largest command sign in view.
3. **ACT:** Require the same sign for four consecutive frames, then print its
   controller command. A connected robot can use the same command interface.

| Sign name | Command |
|---|---|
| `stop` | `S` |
| `no_left_turn` or `turn_left` | `L` |
| `no_right_turn` or `turn_right` | `R` |
| `go_ahead` | `F` |
| `no_entry` | `N` |

## Hardware and software

**Hardware:** Windows/Linux computer, USB or built-in webcam (or a phone
camera stream), and the robot/controller for physical actuation.

**Software:** Python 3.10+, OpenCV, NumPy, Ultralytics, Matplotlib, and
PySerial. Exact dependency constraints are in
[`requirements.txt`](requirements.txt).

## Repository contents

```text
CV_robot/
├── dataset/          Dataset source, split, class, and access record
├── demonstration/    Photos, screenshots, videos, and evidence index
├── models/           Submitted YOLO weights and model notes
├── results/          Metrics, predictions, and final test evidence
├── src/
│   ├── robot/        Hardware command integration notes
│   ├── testing/      Live camera inference
│   ├── training/     Training/evaluation notebooks
│   └── vision/       Vision pipeline notes
└── requirements.txt
```

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the final system

Run the default model with the default camera:

```powershell
python src\testing\testing.py
```

Useful alternatives:

```powershell
python src\testing\testing.py 1
python src\testing\testing.py --list
python src\testing\testing.py --weights models\4th_best.pt
python src\testing\testing.py --rotate 90
python src\testing\testing.py "http://192.168.1.23:8080/video"
```

The phone and computer must be on the same network for an IP-camera URL.
Press `q` to quit, `f` to toggle fullscreen, `m` to switch scan/centre mode,
and `r` to rotate the preview.

## Dataset and training

The training notebooks in [`src/training/`](src/training/) document the
experiments, preprocessing, training, and evaluation workflow. The dataset
access record, class list, annotation format, and split counts must be kept in
[`dataset/README.md`](dataset/README.md). The dataset archive itself is not
included in this checkout; add an approved link or private instructor access
before final submission.

## Model and results

Model usage and checkpoint roles are documented in
[`models/README.md`](models/README.md). Final precision, recall, mAP,
confusion matrix, sample predictions, and Test 1/Test 2 evidence belong in
[`results/README.md`](results/README.md) and [`results/`](results/).

The required robot photos, screenshots, videos, and SEE -> DECIDE -> ACT
evidence are indexed in [`demonstration/README.md`](demonstration/README.md).
No unsupported metrics or demonstration links are claimed by this README;
they should be added when the final test artifacts are available.

## Reproducibility checklist

- [ ] Dataset source, access method, classes, and train/validation/test counts
      recorded
- [ ] Final checkpoint and model task/classes recorded
- [ ] Evaluation metrics and sample predictions added
- [ ] Figure-8 line detection/tracking video added for Test 1
- [ ] Traffic-sign specialization video added for Test 2
- [ ] Robot photos and complete SEE -> DECIDE -> ACT evidence added