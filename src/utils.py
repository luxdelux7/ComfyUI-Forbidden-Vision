import os
import torch
import numpy as np
import cv2
import folder_paths
import yaml
import comfy.model_management as model_management

DEFAULT_CONFIG = {
    'preferred_model_symbol': '⭐',
    'preferred_models': {
        'upscaler': [
            "4xArtFaces_realplksr_dysample.pth",
            "4x_NMKD-Siax_200k.pth",
            "4x-UltraSharp.pth",
            "4x_foolhardy_Remacri.pth",
            "4x-AnimeSharp.pth"
        ],
        'refiner_upscaler': [
            "2xBHI_small_realplksr_small_pretrain.pth",
            "4x-UltraSharp.pth",
            "4x_NMKD-Siax_200k.pth",
            "RealESRGAN_x4plus.pth",
            "ESRGAN_4x.pth"
        ],
        'bbox_primary': [
            "Anzhc20seg20v2%20y8n.pt",
            "face_yolov8n_v2.pt",
            "face_yolov8m.pt",
            "yolov8x6_animeface.pt",
            "yolov8n-face.pt"
        ],
        'bbox_secondary': [
            "Anzhc20seg20v3%20y11n.pt",
            "face_yolov8m.pt",
            "yolov8x6_animeface.pt", 
            "face_yolov8n_v2.pt",
            "yolov8s-face.pt"
        ],
        'segmentation_primary': [
            "Anzhc20seg20v2%20y8n.pt",
            "sam_vit_l_0b3195.pth",
            "yolov8n-seg.pt",
            "yolov8s-seg.pt"
        ],
        'segmentation_secondary': [
            "Anzhc20seg20v3%20y11n.pt",
            "yolov8s-seg.pt",
            "yolov8m-seg.pt",
            "yolov8l-seg.pt"
        ]
    }
}

def load_model_preferences():

    try:
        plugin_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(plugin_root, "model_choice.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
                
                
            config = DEFAULT_CONFIG.copy()
            if user_config:
                if 'preferred_models' in user_config:
                    for category, models in user_config['preferred_models'].items():
                        if category in config['preferred_models']:
                            config['preferred_models'][category] = models
                            
            return config
        else:
            return DEFAULT_CONFIG
            
    except Exception as e:
        print(f"[Forbidden Vision] Error loading config: {e}, using defaults")
        return DEFAULT_CONFIG

def clean_model_name(model_name):

    if not model_name:
        return model_name
    
    symbols = ['⭐', '🌟', '✨', '💎', '🔥', '⚡', '🎯']
    cleaned = model_name
    for symbol in symbols:
        if cleaned.startswith(symbol + ' '):
            cleaned = cleaned[len(symbol) + 1:]
    
    return cleaned
def check_for_interruption():
    if model_management.processing_interrupted():
        raise model_management.InterruptProcessingException()
    
def ensure_model_directories():
    try:
        import os
        
        required_dirs = [
            os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
            os.path.join(folder_paths.models_dir, "ultralytics", "face"),
            os.path.join(folder_paths.models_dir, "ultralytics", "segm"),
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
            os.path.join(folder_paths.models_dir, "ultralytics", "face"),
            os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
            os.path.join(folder_paths.models_dir, "ultralytics", "segm"),
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
    config = load_model_preferences()
    preferred_symbol = config['preferred_model_symbol']
    preferred_bbox_primary = config['preferred_models']['bbox_primary']
    preferred_bbox_secondary = config['preferred_models']['bbox_secondary']
    
    search_paths = [
        os.path.join(folder_paths.models_dir, "ultralytics", "face"),
        os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
        os.path.join(folder_paths.models_dir, "ultralytics"),
        os.path.join(folder_paths.models_dir, "yolo"),
    ]
    
    face_models = set()
    other_models = set()
    
    for path in search_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".pt"):
                    if "ultralytics/face" in path:
                        face_models.add(f)
                    else:
                        other_models.add(f)
    
    final_list = []
    
    face_models_sorted = sorted(list(face_models))
    for model in face_models_sorted:
        final_list.append(model)
    
    all_preferred_bbox = list(preferred_bbox_primary) + list(preferred_bbox_secondary)
    
    for preferred_model in all_preferred_bbox:
        if preferred_model in other_models and preferred_model not in face_models:
            final_list.append(f"{preferred_symbol} {preferred_model}")
            other_models.remove(preferred_model)
    
    remaining_other = sorted(list(other_models - face_models))
    final_list.extend(remaining_other)

    return final_list if final_list else ["None Found"]

def get_sam_models():
    config = load_model_preferences()
    preferred_symbol = config['preferred_model_symbol']
    preferred_seg_primary = config['preferred_models']['segmentation_primary']
    preferred_seg_secondary = config['preferred_models']['segmentation_secondary']
    
    face_paths = [os.path.join(folder_paths.models_dir, "ultralytics", "face")]
    sam_paths = [
        os.path.join(folder_paths.models_dir, "sams"),
        os.path.join(folder_paths.models_dir, "sam"),
        os.path.join(folder_paths.models_dir, "segment_anything"),
    ]
    yolo_seg_paths = [os.path.join(folder_paths.models_dir, "ultralytics", "segm")]
    
    face_models = set()
    for path in face_paths:
        if os.path.isdir(path):
            face_models.update(f for f in os.listdir(path) if f.endswith(".pt"))
            
    yolo_seg_models = set()
    for path in yolo_seg_paths:
        if os.path.isdir(path):
            yolo_seg_models.update(f for f in os.listdir(path) if f.endswith(".pt"))

    sam_models = set()
    for path in sam_paths:
        if os.path.isdir(path):
            sam_models.update(f for f in os.listdir(path) if f.endswith(".pth"))

    final_list = ["None"]
    
    all_preferred_seg = list(preferred_seg_primary) + list(preferred_seg_secondary)
    for face_model in sorted(list(face_models)):
        if face_model in all_preferred_seg:
            final_list.append(f"{preferred_symbol} {face_model}")
        else:
            final_list.append(face_model)
    
    available_models = (yolo_seg_models | sam_models) - face_models
    
    for preferred_model in all_preferred_seg:
        if preferred_model in available_models:
            final_list.append(f"{preferred_symbol} {preferred_model}")
            available_models.discard(preferred_model)
    
    remaining_yolo_seg = sorted([m for m in available_models if m in yolo_seg_models])
    remaining_sam = sorted([m for m in available_models if m in sam_models])
    
    final_list.extend(remaining_yolo_seg)
    final_list.extend(remaining_sam)

    return final_list

def get_yolo_seg_models_only():
    config = load_model_preferences()
    preferred_symbol = config['preferred_model_symbol']
    preferred_seg_secondary = config['preferred_models']['segmentation_secondary']
    
    face_paths = [os.path.join(folder_paths.models_dir, "ultralytics", "face")]
    yolo_seg_paths = [os.path.join(folder_paths.models_dir, "ultralytics", "segm")]
    
    face_models = set()
    for path in face_paths:
        if os.path.isdir(path):
            face_models.update(f for f in os.listdir(path) if f.endswith(".pt"))

    yolo_seg_models = set()
    for path in yolo_seg_paths:
        if os.path.isdir(path):
            yolo_seg_models.update(f for f in os.listdir(path) if f.endswith(".pt"))

    final_list = ["None"]
    
    for face_model in sorted(list(face_models)):
        if face_model in preferred_seg_secondary:
            final_list.append(f"{preferred_symbol} {face_model}")
        else:
            final_list.append(face_model)
    
    available_models = yolo_seg_models - face_models
    
    for preferred_model in preferred_seg_secondary:
        if preferred_model in available_models:
            final_list.append(f"{preferred_symbol} {preferred_model}")
            available_models.discard(preferred_model)
    
    final_list.extend(sorted(list(available_models)))

    return final_list

def get_ordered_upscaler_model_list():
    config = load_model_preferences()
    preferred_symbol = config['preferred_model_symbol']
    preferred_upscaler_models = config['preferred_models']['upscaler']
    
    fast_options = [
        "Fast 4x (Bicubic AA)",
        "Fast 4x (Lanczos)"
    ]
    
    all_models = folder_paths.get_filename_list("upscale_models") or []
    
    final_list = fast_options.copy()
    
    if all_models:
        remaining_models = set(all_models)
        for preferred_model in preferred_upscaler_models:
            if preferred_model in remaining_models:
                final_list.append(f"{preferred_symbol} {preferred_model}")
                remaining_models.remove(preferred_model)
        
        final_list.extend(sorted(list(remaining_models)))
    
    return final_list

def get_refiner_upscaler_models():
    config = load_model_preferences()
    preferred_symbol = config['preferred_model_symbol']
    preferred_refiner_models = config['preferred_models']['refiner_upscaler']
    
    all_models = folder_paths.get_filename_list("upscale_models") or []
    
    final_list = ["Simple: Bicubic (Standard)"]
    
    if all_models:
        remaining_models = set(all_models)
        for preferred_model in preferred_refiner_models:
            if preferred_model in remaining_models:
                final_list.append(f"{preferred_symbol} {preferred_model}")
                remaining_models.remove(preferred_model)
        
        final_list.extend(sorted(list(remaining_models)))
    
    return final_list