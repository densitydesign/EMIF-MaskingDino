#THIS IS SEGM_COMBINED_BW_UNSTABLE.PY, currently its the state-of-the-art

#Add export PYTORCH_ENABLE_MPS_FALLBACK=1 if bugging w/ mps

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



################## WELCOME TO THE ACTUAL TOOL ##################


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Your current device is:", device)

TEXT_THRESHOLD = 0.35
global_folder = 'global_folder'  #substitute with your root path
model_folder = 'model_folder'

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def createBoxes(image_path, text_prompt, box_threshold, token_spans=None):
    print("You are using a threshold of:", box_threshold)
    print("You are using a prompt:", text_prompt)

    model = load_model("weights/GroundingDINO_SwinB_cfg.py",
                       f"{model_folder}/groundingdino_swinb_cogcoor.pth",
                       device=device)  

    image_source, image = load_image(image_path)
    print(f"Debug: Loaded image {image_path}, shape: {image_source.shape}, dtype: {image_source.dtype}")

    boxes_filt, pred_phrases = get_grounding_output(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=TEXT_THRESHOLD,
        cpu_only=device,
        token_spans=token_spans
    )

    print("pred_phrases", pred_phrases)
    #print(f"Debug: Generated boxes, count: {len(boxes_filt)}, phrases: {pred_phrases}")

    # Convert bounding box coordinates to xyxy format
    h, w, _ = image_source.shape
    boxes_filt = boxes_filt.cpu()

    boxes_xyxy = boxes_filt * torch.tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    #print(f"Debug: boxes_xyxy {boxes_xyxy.shape}, {boxes_xyxy}")
    return boxes_xyxy, image_source   

def extractImages(boxes_xyxy, image_path, text_prompt,
                  output_folder,
                  bypass_filling = False,
                  ):
    
    sam_checkpoint = f"{model_folder}/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "mps"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load and set the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    if boxes_xyxy.size == 0:
        print(f"No boxes found for image {image_path}. Printing null box.")
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_output_folder = os.path.join(output_folder, base_filename)
        os.makedirs(image_output_folder, exist_ok=True)
        prompt_word = next((word for word in text_prompt.split() if len(word) > 3), "prompt")
        bw_mask = np.zeros((2048, 2048), dtype=np.uint8)
        bw_mask_output_path = os.path.join(image_output_folder, f"NULL_{base_filename}_{prompt_word}_mask.png")
        cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("NULL mask saved to:", bw_mask_output_path)
        return

    input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

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

    if bypass_filling:
        bw_mask = grayscale_mask.astype(np.uint8)
        
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

    #Measure mask
    height, width = bw_mask.shape[:2]
    print("Final Mask dimensions:", width, "x", height)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a dedicated folder for each image
    image_output_folder = os.path.join(output_folder, base_filename)
    os.makedirs(image_output_folder, exist_ok=True)

    # Extract the first word with more than three characters from TEXT_PROMPT
    prompt_word = next((word for word in text_prompt.split() if len(word) > 3), "prompt")

    # Save the B/W mask image
    bw_mask_output_path = os.path.join(image_output_folder, f"{base_filename}_{prompt_word}_mask.png")
    cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    print("B/W mask saved to:", bw_mask_output_path)

def get_last_processed_image(log_file):
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                return last_line
    except FileNotFoundError:
        return None

def process_images(root_folder, output_folder, start_from_zero=True):
    # Import text prompts from external file
    from text_prompts import text_prompts

    # If start_from_zero is True, erase the log file
    if start_from_zero:
        open(log_file, 'w').close()
        last_processed_image = None
    else:
        last_processed_image = get_last_processed_image(log_file)
    
    # Ensure the log file exists
    if last_processed_image is None:
        open(log_file, 'a').close()

    # Count the total number of images first
    total_images = sum(len(files)
        for _, _, files in os.walk(root_folder)
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')) for file in files))

    with tqdm(total=total_images * len(text_prompts), desc="Processing Images") as pbar:
        for text_prompt, box_threshold in text_prompts.items():  # Get threshold value for each prompt
            for subdir, _, files in os.walk(root_folder):
                files.sort()  # Sort files in alphabetical order
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                        input_image_path = os.path.join(subdir, file)

                        # Skip files that have already been processed
                        if last_processed_image and input_image_path <= last_processed_image:
                            pbar.update(1)
                            continue

                        relative_path = os.path.relpath(subdir, root_folder)
                        output_subfolder = os.path.join(output_folder, relative_path)
                        
                        try:
                            # Create boxes and extract images
                            boxes_xyxy, annotated_frame = createBoxes(input_image_path, text_prompt, box_threshold)  # Pass the threshold value
                            extractImages(boxes_xyxy, input_image_path, text_prompt, output_subfolder)
                            
                            # Log the processed image path
                            logging.info(input_image_path)
                        except UnidentifiedImageError:
                            print(f"Cannot identify image file {input_image_path}. Skipping.")
                        except Exception as e:
                            print(f"Error processing {input_image_path}: {e}")
                        pbar.update(1)


# Define root folder for input images and output folder for results
log_file = f'{global_folder}/process_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
root_folder = f'{global_folder}/INPUT_IMAGES' #FILE INPUT
output_folder = f'{global_folder}/OUTPUT_MASKS' #MAKS OUTPUT

# Process images with refinement enabled
process_images(root_folder, output_folder, start_from_zero=True)

