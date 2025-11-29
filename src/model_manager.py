import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import folder_paths
import comfy.model_management as model_management
from .utils import check_for_interruption
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

class ForbiddenVisionModelManager:
    _instance = None
    _models = {}
    
    MODELS_CONFIG = {
        'face_detect': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models', 
            'filename': 'ForbiddenVision_face_detect_v1.pt',
            'model_type': 'yolo'
        },
        'face_segment': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models',
            'filename': 'ForbiddenVision_face_segment_v1.pth',
            'model_type': 'unetplusplus'
        }
    }
    
    YOLO_DETECTION_SIZE = 640
    FACE_PROCESSING_SIZE = 512
    TARGET_FACE_HEIGHT = int(FACE_PROCESSING_SIZE * 0.75)
    MAX_FACE_WIDTH = int(FACE_PROCESSING_SIZE - 220)
    
    def __init__(self):
        self.models_dir = os.path.join(folder_paths.models_dir, "forbidden_vision")
        os.makedirs(self.models_dir, exist_ok=True)
        self.segmentation_model = None
        self.segmentation_preprocessing = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _download_model(self, model_key):
        try:
            config = self.MODELS_CONFIG[model_key]
            local_path = os.path.join(self.models_dir, config['filename'])
            
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size > 1000:
                    return local_path
                else:
                    print(f"ForbiddenVision: Removing corrupted file: {local_path}")
                    os.remove(local_path)
            
            print(f"ForbiddenVision: Downloading {model_key} from HuggingFace...")
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=config['repo_id'],
                        filename=config['filename'],
                        local_dir=self.models_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )
                    
                    if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 1000:
                        print(f"ForbiddenVision: Successfully downloaded {model_key}")
                        return downloaded_path
                    else:
                         if os.path.exists(downloaded_path):
                            os.remove(downloaded_path)
                            
                except Exception as download_error:
                    print(f"ForbiddenVision: Download attempt {attempt + 1} failed for {model_key}: {download_error}")
                    
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except:
                            pass
            return None
                
        except Exception as e:
            print(f"ForbiddenVision: Critical error downloading {model_key}: {e}")
            return None
    
    def validate_model_availability(self):
        status = {
            'face_detection': False,
            'face_segmentation': False
        }
        
        config = self.MODELS_CONFIG['face_detect']
        local_path = os.path.join(self.models_dir, config['filename'])
        status['face_detection'] = os.path.exists(local_path) and os.path.getsize(local_path) > 1000
        
        config = self.MODELS_CONFIG['face_segment']
        local_path = os.path.join(self.models_dir, config['filename'])
        status['face_segmentation'] = os.path.exists(local_path) and os.path.getsize(local_path) > 1000
     
        return status

    def load_face_detection_model(self):
        model_name = 'ForbiddenVision_face_detect_yolov11s_v1.pt'
        
        if model_name in self._models:
            return self._models[model_name]
        
        config = self.MODELS_CONFIG['face_detect']
        local_path = os.path.join(self.models_dir, config['filename'])
        
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = self._download_model('face_detect')
        
        if not model_path:
            print(f"ForbiddenVision: Could not find face detection model")
            return None
        
        try:
            device = model_management.get_torch_device()
            model = YOLO(model_path)
            model.to(device)
            self._models[model_name] = model
            print(f"ForbiddenVision: Loaded face detection model: {os.path.basename(model_path)}")
            return model
        except Exception as e:
            print(f"ForbiddenVision: Error loading face detection model: {e}")
            return None
        
    def initialize_default_models(self):
        print("ForbiddenVision: Checking default models...")
        
        required_models = {
            'face_detect': 'Face Detection (Small)', 
            'face_segment': 'Face Segmentation'
        }
        
        download_results = {}
        successful_downloads = 0
        
        for model_key, display_name in required_models.items():
            try:
                config = self.MODELS_CONFIG[model_key]
                local_path = os.path.join(self.models_dir, config['filename'])
                
                if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                    print(f"  ✓ {display_name} (cached)")
                    download_results[model_key] = True
                    successful_downloads += 1
                    continue
                
                print(f"  ⏳ Downloading {display_name}...")
                model_path = self._download_model(model_key)
                
                if model_path:
                    print(f"  ✓ {display_name} (downloaded)")
                    download_results[model_key] = True
                    successful_downloads += 1
                else:
                    print(f"  ✗ {display_name} (failed)")
                    download_results[model_key] = False
                    
            except Exception as e:
                print(f"  ✗ {display_name} (error: {e})")
                download_results[model_key] = False
        
        total_models = len(required_models)
        print(f"ForbiddenVision: {successful_downloads}/{total_models} models ready")
        
        return download_results
    
    def load_segmentation_model(self):
        check_for_interruption()
        
        if self.segmentation_model is not None:
            return self.segmentation_model
        
        try:
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            
            config = self.MODELS_CONFIG['face_segment']
            model_path = os.path.join(self.models_dir, config['filename'])
            
            if not os.path.exists(model_path):
                print(f"Segmentation model not found at: {model_path}. Downloading...")
                model_path = self._download_model('face_segment')
                if not model_path:
                    return None
            
            device = model_management.get_torch_device()
            
            model = smp.UnetPlusPlus(
                encoder_name="tu-tf_efficientnetv2_s.in21k_ft_in1k",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                decoder_channels=(256, 128, 64, 32, 16)
            )
            
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            model = model.half().to(device).eval() 
            
            self.segmentation_model = model
            self.segmentation_preprocessing = get_preprocessing_fn(
                "tu-tf_efficientnetv2_s.in21k_ft_in1k",
                pretrained="imagenet"
            )
            
            print("Successfully loaded face segmentation model (Unet++ EfficientNetV2-S)")
            return model
            
        except ImportError as e:
            print(f"Failed to import segmentation_models_pytorch: {e}")
            print("Install with: pip install segmentation-models-pytorch timm")
            return None
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            return None

    def resize_image_for_yolo(self, image_rgb):
        h, w = image_rgb.shape[:2]
        scale = self.YOLO_DETECTION_SIZE / max(h, w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = self.YOLO_DETECTION_SIZE - new_h
        pad_w = self.YOLO_DETECTION_SIZE - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return padded, scale, (left, top)
    
    def scale_bbox_back(self, bbox, scale, offset):
        x1, y1, x2, y2 = bbox
        left_offset, top_offset = offset
        
        x1 -= left_offset
        y1 -= top_offset
        x2 -= left_offset
        y2 -= top_offset
        
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def calculate_face_crop_region(self, face_bbox, img_width, img_height):
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        face_center_x = (face_x1 + face_x2) / 2
        face_center_y = (face_y1 + face_y2) / 2
        
        height_scale = self.TARGET_FACE_HEIGHT / face_height
        width_scale = self.MAX_FACE_WIDTH / face_width
        scale_factor = min(height_scale, width_scale)
        
        crop_size = self.FACE_PROCESSING_SIZE / scale_factor
        
        crop_x1 = face_center_x - crop_size / 2
        crop_y1 = face_center_y - crop_size / 2
        crop_x2 = face_center_x + crop_size / 2
        crop_y2 = face_center_y + crop_size / 2
        
        return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2), scale_factor
    
    def extract_crop_with_padding(self, image, crop_coords):
        img_height, img_width = image.shape[:2]
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        canvas = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
        
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img_width, crop_x2)
        src_y2 = min(img_height, crop_y2)
        
        dst_x1 = max(0, -crop_x1)
        dst_y1 = max(0, -crop_y1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
        canvas = cv2.resize(canvas, (self.FACE_PROCESSING_SIZE, self.FACE_PROCESSING_SIZE), interpolation=cv2.INTER_LINEAR)
        
        return canvas
    
    def segment_face(self, face_crop_rgb):
        check_for_interruption()
        
        seg_model = self.load_segmentation_model()
        if seg_model is None:
            return None
        
        try:
            device = model_management.get_torch_device()
            
            if face_crop_rgb.dtype != np.uint8:
                face_crop_rgb = (face_crop_rgb * 255).astype(np.uint8)
            
            original_h, original_w = face_crop_rgb.shape[:2]
            
            input_image = cv2.resize(face_crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            if self.segmentation_preprocessing:
                input_image = self.segmentation_preprocessing(input_image)
            else:
                input_image = input_image.astype(np.float32) / 255.0
            
            input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).half().to(device)
            
            with torch.no_grad():
                output = seg_model(input_tensor)
                mask_pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
            mask_resized = cv2.resize(mask_pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            return mask_binary
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in face segmentation: {e}")
            return None
    
    def create_oval_mask(self, bbox, h, w):
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        return mask
    
    def clear_cache(self):
        self._models.clear()
        self.segmentation_model = None
        print("Cleared ForbiddenVision model cache")