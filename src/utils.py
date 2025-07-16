import os
import torch
import numpy as np
import cv2
import folder_paths
import comfy.model_management as model_management

def check_for_interruption():
    if model_management.processing_interrupted():
        raise model_management.InterruptProcessingException()
def ensure_model_directories():

    try:
        import os
        
        required_dirs = [
            os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
            os.path.join(folder_paths.models_dir, "sams"),
            os.path.join(folder_paths.models_dir, "ultralytics"),
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
        return True
        
    except Exception as e:
        print(f"Error creating model directories: {e}")
        return False
def find_model_path(model_name, model_type):
    if not model_name or model_name == "None Found":
        return None
    
    search_paths_map = {
        'yolo': [
            os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
            os.path.join(folder_paths.models_dir, "ultralytics"),
            os.path.join(folder_paths.models_dir, "yolo"),
            folder_paths.models_dir
        ],
        'sam': [
            os.path.join(folder_paths.models_dir, "sams"),
            os.path.join(folder_paths.models_dir, "segment_anything"),
        ]
    }
    
    search_paths = search_paths_map.get(model_type, [])
    for path in search_paths:
        potential_path = os.path.join(path, model_name)
        if os.path.exists(potential_path):
            return potential_path
            
    generic_path = folder_paths.get_full_path(model_type, model_name)
    if generic_path:
        return generic_path
        
    return None

def get_yolo_models():
    search_paths = [
        os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
        os.path.join(folder_paths.models_dir, "ultralytics"),
        os.path.join(folder_paths.models_dir, "yolo"),
    ]
    recommended = ["face_yolov8n_v2.pt", "face_yolov8m.pt", "yolov8x6_animeface.pt"]
    found_models = set()
    
    for path in search_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".pt"):
                    found_models.add(f)
    
    final_list = [model for model in recommended if model in found_models]
    remaining = sorted(list(found_models - set(final_list)))
    final_list.extend(remaining)

    return final_list if final_list else ["None Found"]

def get_sam_models():
    search_paths = [
        os.path.join(folder_paths.models_dir, "sams"),
        os.path.join(folder_paths.models_dir, "sam"),
        os.path.join(folder_paths.models_dir, "segment_anything"),
    ]
    
    recommended = [
        "sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"
    ]
    found_models = set()
    
    for path in search_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".pth"):
                    found_models.add(f)
    
    final_list = [model for model in recommended if model in found_models]
    remaining = sorted(list(found_models - set(final_list)))
    final_list.extend(remaining)

    return final_list if final_list else ["None Found"]

def get_ordered_upscaler_model_list():

    preferred_models = [
        "4x_NMKD-Siax_200k.pth",
        "4x-UltraSharp.pth",
        "4x_foolhardy_Remacri.pth",
        "4x-AnimeSharp.pth"
    ]
    
    all_models = folder_paths.get_filename_list("upscale_models") or []
    if not all_models:
        return ["None Found"]
    
    found_preferred = []
    # Use a set for faster lookups of already added models
    found_preferred_set = set()

    for preferred in preferred_models:
        for model in all_models:
            # Check if the preferred name is a substring of the model name
            if preferred in model:
                if model not in found_preferred_set:
                    found_preferred.append(model)
                    found_preferred_set.add(model)

    remaining_models = [model for model in all_models if model not in found_preferred_set]
    
    # Combine the lists and return
    ordered_models = found_preferred + sorted(remaining_models)
    
    return ordered_models if ordered_models else ["None Found"]