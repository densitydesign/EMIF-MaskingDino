import os
import shutil
import subprocess
from tqdm import tqdm
from PIL import Image

def compress_image(image_path, output_path, quality=70):
    try:
        if image_path.lower().endswith('.png'):
            # Use pngquant for PNG images
            output_path = os.path.splitext(output_path)[0] + ".png"
            subprocess.run(['pngquant', '--quality', f'{quality}-{quality}', '--output', output_path, '--force', image_path], check=True)
        else:
            # Use PIL for JPEG images
            img = Image.open(image_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(output_path, 'JPEG', optimize=True, quality=quality)
    except Exception as e:
        print(f"Could not compress {image_path}. Error: {e}")

def compress_folder(input_folder, output_folder, quality=70):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Count total files for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(input_folder)])
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(input_folder):
            relative_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for file in files:
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file)

                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    compress_image(input_file, output_file, quality)
                else:
                    # Copy non-image files
                    shutil.copy2(input_file, output_file)
                
                pbar.update(1)

if __name__ == "__main__":
    input_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/UPSCALED_IMAGES"
    output_folder = "/Users/tommasoprinetti/Documents/DENSITY_OFFICE/EMIF/COMPRESSED_FINAL_IMAGES"
    quality = 70

    compress_folder(input_folder, output_folder, quality)
