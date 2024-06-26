# EMIF MASKINGDINO

## This tool is called MaskingDino and it's meant to produce BW binary masks / straight cut-out based on a semantic prompt

This repo is meant to highlight how to combine GroundingDino and Segment-Anything official repositories to create a unified system able to produce a boolean mask based on a textual prompt. The provided code is a working demo that can be implemented as a tool.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Preparation](#groundingdino-preparation)
- [Text Prompts](#text-prompts)
- [Features](#features)
- [Acknowledgements](#acknowledgements)

## Introduction

GroundingDino is a tool meant to produce boxes and relevations on images based on textual prompts. As of now, I managed to make it work with single words only rather than combinations of them, but that's upcoming. Segment-Anything by Meta is a tool to make segmentations on images. The combination of these two works effectively as SAM can take boxes as inputs and detect the corresponding objects inside them.

The transformers code has been updated following the tweak in this medium article in order to make it work with MPS devices (Apple silicon).

## Installation

MaskingDino has a pretty simple installation. We recommend using `venv` or `conda env` to isolate it.
Disclaimer: Everything was tested on Apple Silicon Macs, we don't know the performances on Windows or non-ARM environments.

1. Clone the GitHub repository:
    ```bash
    cd yourfolder_path
    git clone https://github.com/densitydesign/EMIF-MaskingDino.git
    ```
2. Create the virtual environment with Python >= 3.9.
3. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
4. Prepare the folder with images, change your `global_folder` path in the `PROMPT_TO_MASK_hq.py`.
5. Run the script.

## GroundingDino Preparation

Firstly, we define our `global_folder`. The `global_folder` is the one containing everything, including the folder of inputs and the folder of outputs.

The `model_folder` is an external folder containing the models for the whole process; it's not integrated into the repo as it's definitely too heavy.

### Preparing the Folder

```python
global_folder = "your_global_path"  # substitute with your root path
model_folder = 'your_model_folder'

root_folder = f'{global_folder}/DB_SD_IMAGES'  # FILE INPUT
output_folder = f'/{global_folder}/DEBUG_MASKS'  # MASK OUTPUT
```

Apart from the logfile itself, the script is set up to take images from the `root_folder` and output the results to the `output_folder`. The second one will be created automatically if not yet present, while the root must obviously be present.

As mentioned, the final structure of files will mimic the one of the `root_folder`.

## Text Prompts

As mentioned, the whole process relies on textual prompting for cut-outting. Therefore, in the file `text_prompts.py`, we must organize what we actually want to be detected and cut out.

```python
text_prompts = {
    # "person": 0.25,  # Using the "#" allows to cut off that specified word from the script execution
    "objects": 0.16,
    "clothes": 0.27,
    "hair": 0.25,
    "cap": 0.20,
}
```

This file is just an object containing a list. This list will define what to detect and how precise to be about its detection.

The numbers on the right are the detection threshold; the higher, the stricter. We don't recommend going higher than 0.30 and we recommend keeping values around 0.16-0.25 to provide a wide set of detection possibilities.

The script is prepared to work iteratively on this list of items. So if we have 30 images and 3 text prompts declared, it will run these 3 prompts on these 30 images, producing 90 separate outputs.

## Features

### Working with BW Masks

The main option for this script is to use BW masks as output. While the detections are very precise when it comes to "person", "door", or other simple objects, they are not as precise when cut-outting other entities. We recommend importing these BW masks into a Photoshop file and using them as boolean masks, in order to be fixed and corrected.

### Directly Output PNG

We also provide a `PROMPT_TO_CUT_hq.py` file that outputs a PNG with the applied BW mask. Please be aware that it's an experimental tool and the precision won't be guaranteed.

### Grid Creation Script

Lastly, and this is narrow-specific for the EMIF project, the script `grid_creator.py` creates the 6x6 grids of images with grid gaps ordered by number.

## Acknowledgements

Nothing would have been possible without the following repositories:

- [GroundingDino by IDEA-Lab](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything by Meta](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)
- [SamHQ by Ikeab](https://github.com/SysCV/sam-hq)