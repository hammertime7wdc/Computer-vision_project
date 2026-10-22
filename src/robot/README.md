# Robot integration

This directory is reserved for hardware-specific transport such as serial,
GPIO, or motor-driver code. The current live runner does not send commands to
hardware; it prints the selected command in the terminal so the vision system
can be tested safely without a connected robot.

The sign-to-command mapping and the `SEE -> DECIDE -> ACT` decision logic are
currently in [`../testing/testing.py`](../testing/testing.py). When transport
code is added, document the controller, port/protocol, wiring assumptions,
command format, and safe-stop behavior here.
