# Testing and live inference

[`testing.py`](testing.py) is the final live camera entry point. Run it from
the repository root:

```powershell
python src\testing\testing.py
```

The script accepts a webcam index or phone stream URL as its optional
positional source. Use `--weights` to select a checkpoint, `--rotate` with
`0`, `90`, `180`, or `270` for camera orientation, and `--list` to probe
webcam indexes.

The runner supports both YOLO detection and classification checkpoints. It
filters predictions at confidence `0.80`, selects the largest command sign,
and requires four consistent frames before printing a command. Keyboard
controls are `q` (quit), `f` (fullscreen), `m` (scan/centre mode), and `r`
(rotate).
