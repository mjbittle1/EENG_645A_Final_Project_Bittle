import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_json_to_yolo(json_path, image_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Create mapping from category id to 0-indexed class index
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    print("Categories found in JSON:", categories)
    
    # Standard mapping for our yaml
    target_names = ["Aircraft", "Ship", "Car", "Bridge", "Tank", "Harbor"]
    cat_to_yolo_id = {}
    for cat_id, name in categories.items():
        if name in target_names:
            cat_to_yolo_id[cat_id] = target_names.index(name)
        elif name.title() in target_names:
            cat_to_yolo_id[cat_id] = target_names.index(name.title())
        else:
             print(f"Warning: Category {name} not in target list {target_names}.")

    # Create image dict for quick lookup
    images = {img['id']: img for img in data['images']}
    
    # Aggregate annotations by image_id
    img_to_anns = {img['id']: [] for img in data['images']}
    if 'annotations' in data:
        for ann in data['annotations']:
            img_to_anns[ann['image_id']].append(ann)
            
    # Process each image
    print(f"Processing {len(images)} images for {os.path.basename(json_path)}...")
    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        src_img = os.path.join(image_dir, file_name)
        dst_img = os.path.join(output_images_dir, file_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image {src_img} not found.")
            continue
            
        # Write YOLO label file
        label_file = os.path.join(output_labels_dir, os.path.splitext(file_name)[0] + '.txt')
        with open(label_file, 'w') as lf:
            for ann in img_to_anns[img_id]:
                cat_id = ann['category_id']
                if cat_id not in cat_to_yolo_id:
                    continue
                yolo_class_id = cat_to_yolo_id[cat_id]
                
                # COCO bbox: [x_min, y_min, width, height]
                bbox = ann['bbox']
                x_min, y_min, w, h = bbox
                
                # YOLO format: [x_center, y_center, width, height] (normalized 0-1)
                x_center = (x_min + w / 2.0) / img_width
                y_center = (y_min + h / 2.0) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                lf.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_img_dir = os.path.join(base_dir, "SARDet_100K", "JPEGImages")
    
    # Process Train
    convert_coco_json_to_yolo(
        json_path=os.path.join(base_dir, "SARDet_100K", "Annotations", "train.json"),
        image_dir=os.path.join(src_img_dir, "train"),
        output_images_dir=os.path.join(base_dir, "train", "images"),
        output_labels_dir=os.path.join(base_dir, "train", "labels")
    )
    
    # Process Val
    convert_coco_json_to_yolo(
        json_path=os.path.join(base_dir, "SARDet_100K", "Annotations", "val.json"),
        image_dir=os.path.join(src_img_dir, "val"),
        output_images_dir=os.path.join(base_dir, "valid", "images"),
        output_labels_dir=os.path.join(base_dir, "valid", "labels")
    )
    
    # Process Test
    convert_coco_json_to_yolo(
        json_path=os.path.join(base_dir, "SARDet_100K", "test.json"),
        image_dir=os.path.join(src_img_dir, "test"),
        output_images_dir=os.path.join(base_dir, "test", "images"),
        output_labels_dir=os.path.join(base_dir, "test", "labels")
    )
    print("All conversions finished.")
