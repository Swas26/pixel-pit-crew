# Pixel Pit Crew
prototype desktop app that detects F1 brand logos in race footage using YOLOv8, built with Python + tkinter.

## What it does

- Load any race video (MP4, AVI, MOV, MKV)
- Run YOLOv8 object detection frame-by-frame
- Watch the annotated video with a built-in player
- See a live treemap + table of brand occurrences as you scrub through

## Setup

**Requirements:** Python 3.10+

```bash

# 1. Create a virtual environment
python -m venv .venv

# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

> **Note:** On Apple Silicon Macs, inference runs on MPS. On Windows/Linux it uses CPU (for now).  
> GPU (CUDA) support on Windows/Linux can be enabled by installing the appropriate torch build — see [pytorch.org](https://pytorch.org).

## Project structure

```
.
├── app.py          # Main application
├── model/
│   └── best.pt     # Trained YOLOv8 model weights - to be replaced with oat's ver
├── requirements.txt
└── README.md
```

