# RadioTest

A Tkinter test bench for NooElec NESDR / RTL2832U (RTL-SDR) dongles.

## Features

- Detects the device and shows its USB strings, tuner chip and gain table
- Live spectrum and waterfall (click on either to tune there)
- Controls for frequency (with presets), sample rate, gain or AGC, PPM correction, FFT size and bias-tee
- Stream health readout: throughput, dropped blocks, signal level, clipping, DC offset, I/Q balance, strongest peak
- Wideband FM demodulator with audio output
- Self-test:
  - USB data integrity, using the RTL2832 counter test mode (the same check as `rtl_test`)
  - Sample-rate sweep
  - Tuning sweep across the tuner's range
  - Gain response
  - Raw ADC sanity checks

## Setup (Windows)

1. Bind the WinUSB driver to **Bulk-In, Interface (Interface 0)** with [Zadig](https://zadig.akeo.ie/).
2. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

   `pyrtlsdr[lib]` includes the librtlsdr DLLs.
3. Close any other SDR software (only one program can open the dongle at a time), then run:

   ```
   python rtlsdr_tester.py
   ```

## Notes

- The librtlsdr build bundled with `pyrtlsdrlib` crashes inside libusb when using async reads on Windows. The script therefore streams with synchronous reads in a dedicated thread.
- On RTL dongles, sample rates up to 2.88 MS/s are usually lossless; 3.2 MS/s usually drops samples.
- FM audio needs a sample rate that is a multiple of 240 kHz: 0.96, 1.2, 1.44, 1.92, 2.4 or 2.88 MS/s.
