# Robot integration

Place hardware-specific command transport here when it is available (for
example, serial or GPIO code). The current live runner emits the selected
command in its console and keeps the command mapping in
`src/testing/testing.py`, so the computer-vision demonstration remains
usable without connected hardware.
