# EMIF MASKINGDINO 🎭🦖

## This tool is called MaskingDino and it's meant to produce BW binary masks / straight cut-out based on a semantic prompt

This repo is meant to highlight how to combine GroundingDino and Segment-Anything official repositories to create a unified system able to produce a boolean mask based on a textual prompt. The provided code is a working demo that can be implemented as a tool.

# Colab version of this:
![Static Badge](https://img.shields.io/badge/Prompt_to_mask-ultimate?style=flat&logo=googlecolab&logoColor=%23F9AB00&labelColor=%23000000&color=%2300A8F0)

We've released a Colab version of this tool. Usage with GPU is suggested.

<img src="https://media.tenor.com/xJSM2Ky3WpgAAAAM/steve-ballmer-microsoft.gif" width="100" />




## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Preparation](#groundingdino-preparation)
- [Text Prompts](#text-prompts)
- [Features](#features)
- [Acknowledgements](#acknowledgements)

## Script working scheme
![Scheme of the process](https://github.com/densitydesign/EMIF-MaskingDino/assets/33348451/14eac25a-8e80-4edc-9315-30677331b9d9)

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
3. Activate the environment
4. Install the requirements:
    ```bash
    cd EMIF-MASKINGDINO
    pip install -r requirements.txt
    ```
5. Open `PROMPT_TO_MASK_hq.py` and change the variable `global_folder` with the actual folder-path of your project folder.
6. Inside your `global_folder` create a folder with your images, named `INPUT_IMAGES`
7. The final folder structure should be like:

```
global_folder/
|-- INPUT_IMAGES/
    |-- img_1
    |-- img_2
    |-- img_3
    `-- ...
```

8. [Download](https://drive.google.com/file/d/1oefY5ivTvh35CkiGuVMLRLB30PC3Pao0/view?usp=drive_link) the model folder
9. Open `PROMPT_TO_MASK_hq.py` and change the variable `model_folder` with the actual folder-path of your downloaded model folder.
10. Open `text_prompts.py` and update the list of objects you want to extract from images, more info below.
11. Run:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

12. Execute the script by `python PROMPT_TO_MASK_hq.py`

## Text Prompts
As mentioned, the whole process relies on textual prompting for cut-outting.
Therefore, in the file `text_prompts.py`, we must organize what we actually want to be detected and cut out.

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

The script is prepared to work **iteratively** on this list of items. So if we have 30 images and 3 text prompts declared, it will run these 3 prompts on these 30 images, producing 90 separate outputs.

## Features

### Working with BW Masks

The main option for this script is to use BW masks as output. While the detections are very precise when it comes to "person", "door", or other simple objects, they are not as precise when cut-outting other entities. We recommend importing these BW masks into a Photoshop file and using them as boolean masks, in order to be fixed and corrected.

### Directly Output PNG

We also provide a `PROMPT_TO_CUT_hq.py` file that outputs a PNG with the applied BW mask. Please be aware that it's an experimental tool and the precision won't be guaranteed.

### Grid Creation Script

Lastly, and this is narrow-specific for the EMIF project, the script `grid_creator.py` creates the 6x6 grids of images with grid gaps ordered by number.

# The script itself
This part is dedicated to understand how the logic of the script works, in order to reuse it partially or completely and rework it in the future.

### Import and setup

```python
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import warnings
from PIL import UnidentifiedImageError
import sys
from typing import List, Tuple
import logging

# Adding paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'efficientvit')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam-hq')))

from groundingdino.util.inference import load_model, load_image
from groundingdino.util.inference_on_a_image import get_grounding_output
import groundingdino.datasets.transforms as T

from samHq.segment_anything import sam_model_registry, SamPredictor

from torchvision.ops import box_convert

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*annotate is deprecated*")
```

In the first part we are both importing libraries and logics from the various repsoitories that were cloned and reused.
This logics and functions will be used inside our script to run predictions.

## Creating boxes

```python
def createBoxes(image_path, text_prompt, box_threshold, token_spans=None):
```

This function is responsible for predicting the various boxes based on the original img.
We load the model as explained in the groundingDino repo and return the value of the boxes in the xyzy format that's the one supported by SAM.

## Extracting images

```python
def extractImages(boxes_xyxy, image_path, text_prompt, output_folder, bypass_filling = False):
```

We'll go a bit in-depth on the mechanism of this function as its the core of the script.

```python
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
```

Here we are loading the sam model which is not the "normal" one but its the Sam_HQ that should give s better cutouting performances in precision and accuracy.

```python
# Load and set the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
```

Then we load the img using OpenCV library and set the image inside the predictor with the method `predictor.set_image` that we imported previously

```python
input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
```

We convert the boxes that were previously created by the get-grounding-output function, if present, so that they can be effectively used.

```python
with torch.no_grad():
        masks_refined, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            mask_input=None,
            multimask_output=False,
            return_logits=False,
            hq_token_only=False
        )

        masks_refined = masks_refined.cpu().numpy()
        masks_refined = masks_refined.squeeze(1)
        true_false_mask = np.any(masks_refined, axis=0)
        grayscale_mask = true_false_mask.astype(np.uint8) * 255
```

We then predict the cutouts using the predictor method (imported from SamHQ).
The triple output of that method is reduced to one variable, masks_refined.
This same variable is then passed to a series of steps that give us a grayscale mask (lightweight, binary).

```python
else:
        contour, _ = cv2.findContours(grayscale_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(grayscale_mask, [cnt], 0, 255, -1)

        filled_mask_with_contours = grayscale_mask.copy()
        
        # Parameters
        kernel_size = 10
        blur_kernel_size = 5
        
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        filled_mask_with_contours = cv2.morphologyEx(filled_mask_with_contours, cv2.MORPH_OPEN, kernel)
        filled_mask_with_contours = cv2.morphologyEx(filled_mask_with_contours, cv2.MORPH_CLOSE, kernel)
        filled_mask_with_contours = cv2.GaussianBlur(filled_mask_with_contours, (blur_kernel_size, blur_kernel_size), 0)

        filled_mask = cv2.bitwise_not(filled_mask_with_contours)
        final_mask = cv2.bitwise_not(filled_mask)
        bw_mask = final_mask.astype(np.uint8)
```

Then we post-process the mask we produced by making hole filling and morphology operators with OpenCv2.

```python
# Parameters
        kernel_size = 10
        blur_kernel_size = 5
```

These parameters change the morphology operations on the BW image

The rest of the code is execution.

Bye.

## Acknowledgements

Nothing would have been possible without the following repositories:

- [GroundingDino by IDEA-Lab](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything by Meta](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)
- [SamHQ by Ikeab](https://github.com/SysCV/sam-hq)
