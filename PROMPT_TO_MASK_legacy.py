#

import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import warnings
from PIL import UnidentifiedImageError, Image
import sys
from typing import List, Tuple

sys.path.append("..")
sys.path.append("GroundingDino/GroundingDINO/segment-anything/")
sys.path.append("GroundingDino/GroundingDINO/segment-anything/segment_anything/")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'segment-anything')))

from groundingdino.util.inference import load_model, load_image
from torchvision.ops import box_convert
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict
import supervision as sv

import groundingdino.datasets.transforms as T
#from groundingdino.models import build_model
#from groundingdino.util import box_ops
from groundingdino.util.inference_on_a_image import get_grounding_output
#from groundingdino.util.slconfig import SLConfig
#from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
#from groundingdino.util.vl_utils import create_positive_map_from_span

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*annotate is deprecated*")

device = "mps" 
TEXT_PROMPT = "people ."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.15 

class Model:

    def __init__(self, model_config_path: str, model_checkpoint_path: str, device: str = "mps"):
        self.model = load_model(model_config_path=model_config_path, model_checkpoint_path=model_checkpoint_path, device=device).to(device)
        self.device = device

    def predict_with_caption(self, image: np.ndarray, caption: str, box_threshold: float = BOX_THRESHOLD, text_threshold: float = TEXT_THRESHOLD) -> Tuple[sv.Detections, List[str]]:
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(model=self.model, image=processed_image, caption=caption, box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(source_h=source_h, source_w=source_w, boxes=boxes, logits=logits)
        return detections, phrases

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose([
            T.RandomResize([256], max_size=2048),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(source_h: int, source_w: int, boxes: torch.Tensor, logits: torch.Tensor) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

def createBoxes(image_path, token_spans=None):
    model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "weights/groundingdino_swinb_cogcoor.pth",
                       device=device)  

    image_source, image = load_image(image_path)
    print(f"Debug: Loaded image {image_path}, shape: {image_source.shape}, dtype: {image_source.dtype}")

    boxes_filt, pred_phrases = get_grounding_output(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
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

def add_alpha_channel(image):
    # Add alpha channel to the image
    b, g, r = cv2.split(image)
    a = np.where((b == 0) & (g == 0) & (r == 0), 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r, a))

def extractImages(boxes_xyxy, image_path,
                  output_folder,
                  bypass_filling = False):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "mps"

    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # Load and set the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    if boxes_xyxy.size == 0:
        print(f"No boxes found for image {image_path}. Printing null box.")

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        image_output_folder = os.path.join(output_folder, base_filename)
        os.makedirs(image_output_folder, exist_ok=True)

         # Extract the first word with more than three characters from TEXT_PROMPT
        prompt_word = next((word for word in TEXT_PROMPT.split() if len(word) > 3), "prompt")

        bw_mask = np.zeros((2048, 2048), dtype=np.uint8)  
        #bw_mask = np.invert(bw_mask)
        bw_mask_output_path = os.path.join(image_output_folder, f"OLDSAM_{base_filename}_{prompt_word}_mask.png")
        cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("NULL mask saved to:", bw_mask_output_path) 

        return

    input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    #print("first step begin:")
    masks_refined, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        mask_input=None,
        multimask_output=False,
        return_logits=False,
    )

    masks_refined = masks_refined.cpu().numpy()
    masks_refined = masks_refined.squeeze(1)
    true_false_mask = np.any(masks_refined, axis=0)
    grayscale_mask = true_false_mask.astype(np.uint8) * 255

    #print("second step done:", masks_refined)

    if bypass_filling:
        bw_mask = grayscale_mask.astype(np.uint8)
    else:
        
        contour, _ = cv2.findContours(grayscale_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(grayscale_mask, [cnt], 0, 255, -1)

        filled_mask_with_contours = grayscale_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        filled_mask_with_contours = cv2.morphologyEx(filled_mask_with_contours, cv2.MORPH_CLOSE, kernel)
        filled_mask_with_contours = cv2.GaussianBlur(grayscale_mask, (5, 5), 0)
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
    prompt_word = next((word for word in TEXT_PROMPT.split() if len(word) > 3), "prompt")

    # Save the B/W mask image
    bw_mask_output_path = os.path.join(image_output_folder, f"{base_filename}_{prompt_word}_mask.png")
    cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    print("B/W mask saved to:", bw_mask_output_path)


def process_images(root_folder, output_folder, should_refine=True):
    # Count the total number of images first
    total_images = sum(len(files) for _, _, files in os.walk(root_folder) if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files))

    with tqdm(total=total_images, desc="Processing Images") as pbar:
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_image_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(subdir, root_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    
                    try:
                        # Create boxes and extract images
                        boxes_xyxy, annotated_frame = createBoxes(input_image_path)
                        extractImages(boxes_xyxy, input_image_path, output_subfolder)
                    except UnidentifiedImageError:
                        print(f"Cannot identify image file {input_image_path}. Skipping.")
                    except Exception as e:
                        print(f"Error processing {input_image_path}: {e}")
                    pbar.update(1)

# Define root folder for input images and output folder for results
root_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/DEF_MOCKUP_LIBRI/DB_IMMAGINI" #FILE INPUT
output_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/DEF_MOCKUP_LIBRI/DB_MASCHERE" #SALVATAAGGIO MASCHERE

# Process images with refinement enabled
process_images(root_folder, output_folder)
