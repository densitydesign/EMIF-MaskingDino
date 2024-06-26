#THIS SCRIPT IS RESPONSIBLE FOR THE CREATION OF GRIDS

import os
from PIL import Image
from tqdm import tqdm
import re

root_dir = '/Volumes/Cartella pubblica di Tommaso Prinetti/EMIF_CUTOUT/DB_SD_MASCHERE' #CHANGE ACCORDINGLY
output_dir = '/Volumes/Cartella pubblica di Tommaso Prinetti/EMIF_CUTOUT/DB_GRIDS' #CHANGE ACCORDINGLY
grid_gap = 100 #CHANGE ACCORDINGLY

def find_images(root_dir, keywords):
    image_dict = {keyword: [] for keyword in keywords}
    nation_category_dict = {keyword: {} for keyword in keywords}
    nation_pattern = r"(Italy|Croatia|Cyprus|Portugal|Romania|Slovakia|Ukraine|Estonia|Czech|Greece)" #CHANGE ACCORDINGLY
    category_pattern = r"(family|working)" #CHANGE ACCORDINGLY
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                for keyword in keywords:
                    if keyword in file.lower():
                        relative_path = os.path.relpath(subdir, root_dir)
                        nation_match = re.search(nation_pattern, file, re.IGNORECASE)
                        category_match = re.search(category_pattern, file, re.IGNORECASE)
                        if nation_match and category_match:
                            nation = nation_match.group(0).capitalize()
                            category = category_match.group(0).lower()
                            if nation not in nation_category_dict[keyword]:
                                nation_category_dict[keyword][nation] = {}
                            if category not in nation_category_dict[keyword][nation]:
                                nation_category_dict[keyword][nation][category] = []
                            nation_category_dict[keyword][nation][category].append((relative_path, os.path.join(subdir, file)))
                        break
    return nation_category_dict

def create_image_grids(root_dir, keywords, grid_size=(6, 6), output_dir=output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find and log all usable images
    nation_category_dict = find_images(root_dir, keywords)
    for keyword, nations in nation_category_dict.items():
        for nation, categories in nations.items():
            for category, image_paths in categories.items():
                print(f"Found {len(image_paths)} images for nation '{nation}', category '{category}', keyword '{keyword}'")

    # Sort the image paths alphabetically and create grids for each category
    for keyword, nations in nation_category_dict.items():
        for nation, categories in nations.items():
            for category, image_paths in categories.items():
                image_paths.sort(key=lambda x: x[1])  # Sort by file path
                
                if not image_paths:
                    continue  # Skip if there are no images in this category

                for i in tqdm(range(0, len(image_paths), grid_size[0] * grid_size[1]), desc=f"Processing {keyword} images for {nation} - {category}"):
                    grid_images = image_paths[i:i + grid_size[0] * grid_size[1]]

                    # Load images with detailed progress and error handling
                    images = []
                    for relative_path, img_path in tqdm(grid_images, desc=f"Loading {keyword} images", leave=False):
                        try:
                            img = Image.open(img_path).convert('RGBA')
                            images.append((relative_path, img))
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

                    if not images:
                        print(f"No valid images found for {keyword} grid starting at index {i}")
                        continue

                    # Get max width and height
                    max_width = max(img.width for _, img in images)
                    max_height = max(img.height for _, img in images)

                    # Calculate the size of the grid with gaps
                    gap = grid_gap
                    grid_width = max_width * grid_size[0] + gap * (grid_size[0] - 1)
                    grid_height = max_height * grid_size[1] + gap * (grid_size[1] - 1)

                    # Create a transparent background grid with the new dimensions
                    grid_img = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))

                    # Paste images into the grid with gaps
                    for idx, (relative_path, img) in enumerate(images):
                        row = idx // grid_size[0]
                        col = idx % grid_size[0]
                        x_position = col * (max_width + gap)
                        y_position = row * (max_height + gap)
                        grid_img.paste(img, (x_position, y_position))

                    # Save the grid image in the corresponding subfolder structure
                    grid_subdir = os.path.join(output_dir, nation, category)
                    os.makedirs(grid_subdir, exist_ok=True)
                    grid_output_path = os.path.join(grid_subdir, f"{keyword}_grid_{i // (grid_size[0] * grid_size[1]) + 1}.png")
                    try:
                        grid_img.save(grid_output_path)
                        print(f"Saved grid image: {grid_output_path}")
                    except Exception as e:
                        print(f"Error saving grid image {grid_output_path}: {e}")

if __name__ == "__main__":
    keywords = [ 'clothes', 'hair', 'objects', 'person']
    create_image_grids(root_dir, keywords)
