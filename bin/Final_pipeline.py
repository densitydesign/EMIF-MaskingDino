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

from groundingdino.util.inference import load_model, load_image
from torchvision.ops import box_convert
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference_on_a_image import get_grounding_output
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict
import supervision as sv

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*annotate is deprecated*")

device = "mps"
should_refine = True
TEXT_PROMPT = "people ."

class Model:

    def __init__(self, model_config_path: str, model_checkpoint_path: str, device: str = "mps"):
        self.model = load_model(model_config_path=model_config_path, model_checkpoint_path=model_checkpoint_path, device=device).to(device)
        self.device = device

    def predict_with_caption(self, image: np.ndarray, caption: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[sv.Detections, List[str]]:
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(model=self.model, image=processed_image, caption=caption, box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(source_h=source_h, source_w=source_w, boxes=boxes, logits=logits)
        return detections, phrases

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
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

def createBoxes(image_path):
    model = load_model("GroundingDino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "GroundingDino/GroundingDINO/weights/groundingdino_swint_ogc.pth",
                       device=device)
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25   

    image_source, image = load_image(image_path)

    boxes_filt, _ = get_grounding_output(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        cpu_only=device
    )

    # Convert bounding box coordinates to xyxy format
    h, w, _ = image_source.shape
    boxes_filt = boxes_filt.cpu()
    boxes_xyxy = boxes_filt * torch.tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    #print("Debug: boxes_xyxy", boxes_xyxy)
    return boxes_xyxy, image_source

def cutout_original_image(image, masks):
    cutouts = []
    for mask in masks:
        cutout = np.zeros_like(image)
        for c in range(3):
            cutout[:, :, c] = np.where(mask, image[:, :, c], 0)
        cutouts.append(cutout)
    return cutouts

def add_alpha_channel(image):
    # Add alpha channel to the image
    b, g, r = cv2.split(image)
    a = np.where((b == 0) & (g == 0) & (r == 0), 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r, a))

def refine_masks(masks, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined_masks = []
    for mask in masks:
        mask = mask.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closing operation to fill holes
        refined_masks.append(mask)
    return refined_masks

def extractImages(boxes_xyxy, image_path, output_folder, should_refine=should_refine):
    sam_checkpoint = "GroundingDino/GroundingDINO/sam_vit_h_4b8939.pth"
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
        print(f"No boxes found for image {image_path}. Skipping.")
        return

    input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    #print("Debug: input_boxes", input_boxes, input_boxes.shape, type(input_boxes))
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    #print("Debug: transformed_boxes", transformed_boxes, transformed_boxes.shape, type(transformed_boxes))

    # First mask prediction
    _, _, low_res_masks = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output= False,  # Single mask per object
    )

    # Use the low-res masks for a second prediction
    masks_refined, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        mask_input=low_res_masks,  # Use low-res masks from the first prediction
        multimask_output=False,  # Single mask per object
    )

    #print("Debug: masks_refined (second prediction)", masks_refined, masks_refined.shape, type(masks_refined))

    # Convert masks to numpy and refine them if needed
    masks_refined = masks_refined.cpu().numpy()
    masks_refined = masks_refined.squeeze(1)
    #print("Debug: masks_refined (numpy)", masks_refined.shape, type(masks_refined))

    if should_refine:
        refined_masks = refine_masks(masks_refined)
        #print("Debug: refined_masks", refined_masks, len(refined_masks))
    else:
        refined_masks = masks_refined

    # Use refined masks for further processing
    cutouts = cutout_original_image(image, refined_masks)
    
    # Create a transparent canvas
    h, w, _ = image.shape
    transparent_canvas = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Combine cutouts on the transparent canvas
    for cutout in cutouts:
        cutout_bgra = add_alpha_channel(cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        transparent_canvas = cv2.add(transparent_canvas, cutout_bgra)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a dedicated folder for each image
    image_output_folder = os.path.join(output_folder, base_filename)
    os.makedirs(image_output_folder, exist_ok=True)

    # Extract the first word with more than three characters from TEXT_PROMPT
    prompt_word = next((word for word in TEXT_PROMPT.split() if len(word) > 3), "prompt")

    # Save the compounded image
    final_output_path = os.path.join(image_output_folder, f"{prompt_word}.png")
    cv2.imwrite(final_output_path, transparent_canvas)
    print("Compounded image saved to:", final_output_path)
    
def process_images(root_folder, output_folder, should_refine=should_refine):
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
                        extractImages(boxes_xyxy, input_image_path, output_subfolder, should_refine)
                    except UnidentifiedImageError:
                        print(f"Cannot identify image file {input_image_path}. Skipping.")
                    except Exception as e:
                        print(f"Error processing {input_image_path}: {e}")
                    pbar.update(1)

# Define root folder for input images and output folder for results
root_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/TEST"
output_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/TEST_DETECTION"

# Process images with refinement enabled
process_images(root_folder, output_folder, should_refine=should_refine)
