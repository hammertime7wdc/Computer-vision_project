# Computer Vision Robot

## Project description

This project is a real-time computer-vision system for recognizing traffic
signs and converting recognized signs into robot movement commands. The system
uses a camera, OpenCV, and an Ultralytics YOLO model.

The program supports a USB webcam, built-in camera, or phone-camera stream.
The live inference program is located at
[`src/testing/testing.py`](src/testing/testing.py).

## System workflow

The system follows the **SEE -> DECIDE -> ACT** process:

1. **SEE** - Capture a frame from the camera and prepare its orientation and
   size.
2. **DECIDE** - Run YOLO detection or classification, filter low-confidence
   predictions, and smooth results across multiple frames.
3. **ACT** - Select the largest recognized command sign and output its robot
   command.

The command mapping used by the program includes:

| Recognized sign | Command |
|---|---|
| `stop` | `S` |
| `no_left_turn` / `turn_left` | `L` |
| `no_right_turn` / `turn_right` | `R` |
| `go_ahead` | `F` |
| `no_entry` | `N` |

The program waits for a sign to remain consistent over several frames before
emitting a command. This reduces unintended commands from unstable
predictions.

## Repository structure

```text
CV_robot/
├── README.md
├── requirements.txt
├── dataset/
│   └── README.md
├── demonstration/
│   └── README.md
├── models/
│   ├── README.md
│   └── *.pt
├── results/
│   └── README.md
└── src/
    ├── robot/
    ├── testing/
    │   ├── README.md
    │   └── testing.py
    ├── training/
    │   ├── README.md
    │   └── *.ipynb
    └── vision/
```

## Hardware

The system requires:

- A USB webcam, built-in camera, or phone camera stream
- A computer capable of running Python and YOLO inference
- A robot or controller when physical command execution is required

The computer-vision program can be tested without connecting the robot. In
that mode, recognized commands are printed in the terminal.

## Software

- Python 3.10 or newer
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- PySerial

The package versions are listed in
[`requirements.txt`](requirements.txt).

## Installation

Open PowerShell in the project root and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the system

Run the default model with the default camera:

```powershell
python src\testing\testing.py
```

The command must be run from the project root. The correct path is
`src\testing\testing.py`; there is no `robot\testing\testing.py` path in this
repository.

### Camera selection

Use a specific webcam index:

```powershell
python src\testing\testing.py 0
python src\testing\testing.py 1
```

List available camera indexes:

```powershell
python src\testing\testing.py --list
```

Use a phone-camera stream:

```powershell
python src\testing\testing.py "http://192.168.1.23:8080/video"
```

The phone and computer must be connected to the same network.

### Model and camera orientation

The default model is `models\best.pt`. A different model can be selected with
`--weights`:

```powershell
python src\testing\testing.py --weights models\4th_best.pt
```

For a vertically oriented phone camera, set the rotation:

```powershell
python src\testing\testing.py --rotate 90
```

Supported rotation values are `0`, `90`, `180`, and `270`.

### Preview controls

The live preview supports the following keyboard controls:

- `q` - Quit the program
- `f` - Toggle fullscreen
- `m` - Switch between multi-sign scan and centre classification
- `r` - Rotate the camera image

## Trained models

The trained YOLO weights are stored in [`models/`](models/):

- [`best.pt`](models/best.pt)
- `first_best.pt`
- `second_best.pt`
- `3rd_best.pt`
- `4th_best.pt`
- `segunda_etapa_best.pt`

The live program uses `best.pt` by default. It prints the loaded model task
and recognized class names when it starts. Additional model information is
available in [`models/README.md`](models/README.md).

## Dataset

The [`dataset/`](dataset/) folder contains the documentation and access
information for the dataset used by the project. The dataset is kept separate
from the source code and model weights.

The dataset documentation records the source, annotation format, class list,
and training, validation, and testing splits.

## Training and experiments

The project notebooks are stored in
[`src/training/`](src/training/):

- `test_2_robot.ipynb`
- `test_2_robot_again.ipynb`
- `test_4.ipynb`
- `test_4_improvement.ipynb`

These notebooks contain the model-training experiments, evaluation work, and
robot-testing iterations used during development.

## Results and demonstration

The [`results/`](results/) folder is used for prediction outputs, evaluation
metrics, testing logs, and other computer-vision results.

The [`demonstration/`](demonstration/) folder is used for robot photographs,
system screenshots, demonstration videos, and evidence of the complete
SEE -> DECIDE -> ACT process.

## Project information

- **Group members:**
  - Albano, Luis Van Dirk
  - Borac, John Joshua
  - Carullo, John Wilbert
  - Regalado, Roger Jr.
  - Reyes, Reynaldo Jr.
  - Sabio, Jacob Angel

- **Project specialization:**

Traffic Sign Recognition

- **Robot/controller:** The computer-vision application outputs movement
  commands for the connected robot/controller.