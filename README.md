# BOS Blender Simulator

A Blender-based simulation pipeline for generating synthetic Background Oriented Schlieren fields and analyze results.

---

## 🚀 Features

* Python-based simulation pipeline
* Portable project structure (Linux / macOS / Windows)
* Config-driven workflow (`config.toml`)
* Scientific outputs (NumPy / TIFF / etc.)

---

## ⚙️ Requirements

* Packages (install inside Blender Python):

  * numpy
  * tifffile
  * toml

Install example:

```bash
/path/to/blender/python/bin/python -m pip install package-name
```

* Packages for postprocessing:

  * open-cv

---

## ▶️ How to Run

Run Blender (open graphical interface):

```bash
blender --python main_blender.py
```

Or in background mode:

```bash
blender -b --python main_blender.py
```
