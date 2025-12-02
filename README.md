# 🌌 darkpfp-gui — Advanced Dark PFP Processor (PyQt5 + OpenCV)

**Author:** elfloli  
**Version:** 1.0  
**Language:** Python 3  
**GUI:** PyQt5  
**Processing:** OpenCV + Pillow + optional DL models (HED / U²Net)

<p align="center">
  <img src="https://dummyimage.com/900x350/000/ffffff&text=darkpfp+GUI+Preview" />
</p>

---

## ✨ Overview

`darkpfp_gui.py` is a powerful desktop tool for generating **dark-themed neon profile pictures** with highly customizable edge detection, segmentation, neon glow, presets, batch processing, and more.

This GUI version includes:

- Multiple edge detection modes (Fusion / Canny / HED)
- Foreground boosting (saliency / segmentation)
- U²Net + GrabCut support
- Line thinning (skeletonization)
- Soft neon glow renderer
- Color tinting
- Square resizing (pad / stretch)
- INI preset system (save / delete)
- Folder memory (input/output history)
- **NEW: Full folder batch processing**

---

## 🚀 Features

### 🔥 Edge Detection Modes
- **Fusion** — multi-source adaptive edge mixing  
- **Canny** — classic clean edges  
- **HED** *(optional)* — deep learning contours using Caffe model  
  Requires:  
  - `hed_pretrained_bsds.caffemodel`  
  - `deploy.prototxt` or `hed.prototxt`

### 🧠 Foreground / Saliency / Segmentation
- Static saliency (opencv-contrib)
- U²Net segmentation (`u2net.onnx` / `u2netp.onnx`)
- GrabCut refinement
- Foreground-weighted edge boosting
- Adjustable mask intensity

### 🎨 Stylization Modes
- **Dark Neon Profile Picture**
- **Line Art (white on black)**

### 🧩 Image Processing Tools
- Upscale 1× / 2× / 4× before stylization
- Square image generation:
  - Padding (black)
  - Stretching
- Vignette
- Scanlines
- Film grain

### 💾 Presets (INI)
- Create unlimited presets
- Delete presets
- Load automatically on startup
- Stores all style parameters

### 📁 NEW: Batch Processing
- Select a folder
- Automatically processes all supported images
- Saves results using original filenames

---

## 📦 Installation

### 1️⃣ Install required packages
```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python PyQt5 pillow numpy
2️⃣ (Optional) HED model files
Place these in the same directory as darkpfp_gui.py:

Копіювати код
hed_pretrained_bsds.caffemodel
deploy.prototxt   OR   hed.prototxt
3️⃣ (Optional) U²Net segmentation models
Also placed next to the script:

Копіювати код
u2net.onnx
u2netp.onnx
▶️ How to Run
bash
Копіювати код
python darkpfp_gui.py
📁 Project Structure
cpp
Копіювати код
darkpfp_gui/
│
├── darkpfp_gui.py
├── darkpfp_gui.ini
├── hed_pretrained_bsds.caffemodel    (optional)
├── deploy.prototxt                   (optional)
├── u2net.onnx                        (optional)
└── u2netp.onnx                       (optional)
🖼 Sample Output
<p align="center"> <img src="https://dummyimage.com/500x500/000/ffffff&text=Neon+Output+Sample" /> </p> <p align="center"> <img src="https://dummyimage.com/500x500/000/ffffff&text=LineArt+Output+Sample" /> </p>
🔧 Technology Stack
Component	Description
PyQt5	GUI framework
OpenCV	Image processing, HED model loading
Pillow	Image compatibility & saving
U²Net	Foreground segmentation
INI Files	Preset management

📝 Preset System
Presets are stored in:

Копіювати код
darkpfp_gui.ini
Each preset contains:

Edge detection settings

Glow and tint parameters

Segmentation mode

Filters (grain, vignette, scanlines)

UI options

Loaded instantly on startup

🛠 Developer Notes
Efficient Unicode-safe image loading via cv2.imdecode

Fusion edges combine LAB, HSV, LoG, multi-scale Canny, CLAHE

Segmentation pipeline:

U²Net → GrabCut → soft mask blending

Neon glow is generated using multi-layer dilation + Gaussian stacks

❤️ Credits
OpenCV contributors

U²Net authors

HED model authors

GUI & processing logic by elfloli

📜 License
MIT License

⭐ Support This Project
If you like this tool — give it a ⭐ on GitHub!

yaml
Копіювати код

---

If you want, I can also prepare:

🔥 a dark-themed README version  
📌 a version with verified GitHub badges  
🎨 a custom logo  
📁 a ready-to-upload GitHub repository structure  

Just tell me!
