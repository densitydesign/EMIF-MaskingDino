# EMIF MASKINGDINO
## This tool is called MaskingDino and it's meant to produce BW binary masks / straight cut-out based on a semantic prompt

This repo is ment to highlight how to combine GroundingDino and Segment-Anything official repositories to create an unified system able to produce a boolean mask based on a textual prompt. The provided code is a working demo that can be implemented as a tool.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Preparation](#GroundingDino-preparation)
- [Text prompts](#Text-prompts)
- [Features](#working-with-bw-masks--working-with-png)
- [Acknowledgements](#acknowledgements)

## Introduction

GroundingDino is a tool ment to produce boxes and relevation on images based on textual prompts. As of now, I managed to make it work with single-words only rather than combinations of it, that's upcoming. Segment-anything by Meta is instead a tool to make segmentations on images. The combination of these two works effectively as SAM can take boxes as inputs and detect the corresponding objects inside it.

The transformers code has been updated following the tweak in this medium article in order to make it work with MPS devices (Apple silicon).

## Installation

MaskingDino has a pretty simple installation, we recommend sing venv or conda env to isolate it.
Disclaimer: everything was tested on Apple Silicon Macs, we don't know the performances on Windows or non-ARM envs.

1. Clone the github repository:
``` cd yourfolder_path ```
``` git clone https://github.com/densitydesign/EMIF-MaskingDino.git ```
2. Create the virtual env with python >= 3.9
3. Install the requirements:
``` install -r requirements.txt ```
4. Prepare the folder with images, change your global_folder path in the PROMPT_TO_MASK_hq.py
5. Run the script

## GroundingDino preparation

```
global_folder = "your_global_path"  #substitute with your root path
model_folder = 'your_model_folder'
```

Firstly, we define our global_folder. The global_folder is the one containing everything, the folder of inputs and the folder of outputs..

The `model_folder` is an external folder containing the models for the whole process, it's not integrated in the repo as it's definetly too heavy.

### Preparing the folder:

```
root_folder = f'{global_folder}/DB_SD_IMAGES' #FILE INPUT
output_folder = f'/{global_folder}/DEBUG_MASKS' #MAKS OUTPUT
``` 

Apart from the logfile itself, the script is set-up to take images from the root_folder and output the reslts to the output folder. The second one will be created automatically if not yet present, while the root must obviously be present yet.

As yet said, the final structure of files will mimic the one of the root_folder.

### Text prompts

As said, the whole process relies on textual prompting for cut-outing.
Therefore, in the file: text_prompts.py, we must organize what we actually want to be detected and cutouted.

```
text_prompts = {
    #"person": 0.25,   #Using the "#" allows to cut-off that specified work from the script execution
    "objects": 0.16,
    "clothes": 0.27,
    "hair": 0.25,
    "cap": 0.20,
}
```

This file is jst an object containing a list. This list will define what to detect and how precise to be about its detection.

The numbers on the right are the detection threshold, the higher, the stricter. We don't recommend to go higher than 0.30 and we do recommend to keep values around 0.16-0.25 to provide a wide set of detection possibilities.

The script is prepared to work iteratively on this list of items. So if we have 30 images and 3 text_prompts declared, it will run these 3 prompts on these 30 images producing 90 separated outputs.

# Working with BW Masks / Working with PNG

## BW Masks

The main option for this script is to use BW masks as output. As the detection are very precise when it comes to "person", "door", or other simple objects, it's not as precise when cut-outing other entities. We recommend to import these bw masks into a photoshop file and use them as boolean masks, in order to be fixed and corrected.

## Directly output PNG

We provide also a PROMPT_TO_CUT_hq.py file that otputs a png with the applied bw mask. Please be aware that it's an experimental tool and the precision won't be guaranteed.

# Acknowledgements

Nothing would have been possible without the following repositories:

[GroundingDino by IDEA-Lab](https://github.com/IDEA-Research/GroundingDINO)
[SegmentAnything by Meta](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)
[SamHQ by Ikeab](https://github.com/SysCV/sam-hq)