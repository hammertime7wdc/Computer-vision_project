# Vision pipeline

The current demonstrable pipeline is implemented in
[`../testing/testing.py`](../testing/testing.py). It combines camera input,
frame preparation, YOLO inference, confidence filtering, scan-window
non-maximum suppression, visualization, temporal smoothing, and command
selection.

Future reusable vision helpers should be placed in this directory and kept
independent from camera or robot I/O.
