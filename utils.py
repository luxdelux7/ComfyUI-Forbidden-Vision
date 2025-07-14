import os
import torch
import numpy as np
import cv2
import folder_paths
import comfy.model_management as model_management

def check_for_interruption():
    """Check if ComfyUI has requested cancellation"""
    if model_management.processing_interrupted():
        raise model_management.InterruptProcessingException()
def ensure_model_directories():
    """
    Create model directories if they don't exist, following Impact Pack structure
    """
    try:
        import os
        
        required_dirs = [
            os.path.join(folder_paths.models_dir, "ultralytics", "bbox"),
            os.path.join(folder_paths.models_dir, "sams"),
            os.path.join(folder_paths.models_dir, "ultralytics"),
        ]
        
        created_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                created_dirs.append(dir_path)
                print(f"Created model directory: {dir_path}")
        
        if created_dirs:
            print(f"Created {len(created_dirs)} missing model directories")
        else:
            print("All model directories already exist")
            
        return True
        
    except Exception as e:
        print(f"Error creating model directories: {e}")
        return False
def find_model_path(model_name, model_type):
    """
    Finds the full path for a given model name and type ('yolo' or 'sam').
    This is the single source of truth for model locations.
    """
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
    """Scans for YOLO bbox models and returns a prioritized list."""
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
    """Scans for regular SAM models only (no SAM2)."""
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

def safe_tensor_to_numpy(tensor, target_range=(0, 255)):
    """Safely convert tensor to numpy with proper range handling."""
    try:
        if len(tensor.shape) == 4:
            np_array = tensor.squeeze(0)
        else:
            np_array = tensor
        
        np_array = np_array.detach().cpu()
        np_array = torch.clamp(np_array, 0.0, 1.0)
        
        if target_range == (0, 255):
            np_array = (np_array * 255.0).numpy().astype(np.uint8)
        else:
            np_array = np_array.numpy().astype(np.float32)
        
        return np_array
    except Exception as e:
        print(f"Error in tensor conversion: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8 if target_range == (0, 255) else np.float32)

def safe_numpy_to_tensor(np_array, input_range=(0, 255)):
    """Safely convert numpy to tensor with proper range handling."""
    try:
        if input_range == (0, 255):
            tensor = torch.from_numpy(np_array).float() / 255.0
        else:
            tensor = torch.from_numpy(np_array).float()
        
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        return torch.clamp(tensor, 0.0, 1.0)
    except Exception as e:
        print(f"Error in numpy conversion: {e}")
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

def process_mask_morphology(mask, expansion=0, blur_size=0, blur_sigma=1.0):
    try:
        working_mask = mask.copy()
        
        if expansion > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion*2+1, expansion*2+1))
            working_mask = cv2.dilate(working_mask, kernel, iterations=1)

        if blur_size > 0:
            kernel_size = max(3, int(blur_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
            working_mask = cv2.GaussianBlur(working_mask, (kernel_size, kernel_size), blur_sigma)
            
        return np.clip(working_mask, 0.0, 1.0)
    except Exception as e:
        print(f"Error in mask processing: {e}")
        return mask