import os
import torch
import numpy as np
import cv2
import folder_paths
import yaml
import comfy.model_management as model_management
import os
import torch
import numpy as np
import cv2
import folder_paths
import comfy.model_management as model_management


RESOLUTIONS = {
        "📱 1:4 Ultra-Tall (512x2048)": (512, 2048),
        "📱 1:2.4 Vertical (640x1536)": (640, 1536),
        "📱 9:16 Portrait (768x1344)": (768, 1344),
        "📱 2:3 Portrait (832x1216)": (832, 1216),
        "📱 3:4 Portrait (896x1152)": (896, 1152),
        "📱 4:5 Portrait (1024x1280)": (1024, 1280),
        "📱 2:3 SD1.5 (512x768)": (512, 768),

        "🔳 1:1 Square (1024x1024)": (1024, 1024),
        "🔳 1:1 Square (768x768)": (768, 768),
        "🔳 1:1 Square (512x512)": (512, 512),

        "🖼️ 5:4 Landscape (1280x1024)": (1280, 1024),
        "🖼️ 4:3 Landscape (1152x896)": (1152, 896),
        "🖼️ 3:2 Landscape (1216x832)": (1216, 832),
        "🖼️ 3:2 SD1.5 (768x512)": (768, 512),
        "🖼️ 16:9 Widescreen (1344x768)": (1344, 768),

        "🎥 21:9 Cinematic (1536x640)": (1536, 640),
        "🎥 3.8:1 Panoramic (1984x512)": (1984, 512),
        "🎥 4:1 Ultra-Wide (2048x512)": (2048, 512),
    }
def clean_model_name(model_name):
    """Clean model name by removing URL encoding"""
    if model_name:
        return model_name.replace("%20", " ")
    return model_name

def check_for_interruption():
    """Check if processing should be interrupted"""
    try:
        if hasattr(model_management, 'processing_interrupted'):
            if model_management.processing_interrupted():
                raise model_management.InterruptProcessingException()
    except AttributeError:
        pass

def get_refiner_upscaler_models():
    """Get available upscaler models for the refiner"""
    fast_options = [
        "Simple: Bicubic (Standard)"
    ]
    
    all_models = folder_paths.get_filename_list("upscale_models") or []
    
    final_list = fast_options.copy()
    final_list.extend(sorted(all_models))
    
    return final_list

def check_forbidden_vision_models():
    forbidden_vision_dir = os.path.join(folder_paths.models_dir, "forbidden_vision")
    
    model_status = {}
    required_files = [
        "ForbiddenVision_face_detect_v1.pt",
        "ForbiddenVision_face_detect_v1.onnx",
        "ForbiddenVision_face_segment_v1.safetensors",
        "ForbiddenVision_neural_corrector_v1.safetensors"
    ]
    
    for filename in required_files:
        file_path = os.path.join(forbidden_vision_dir, filename)
        model_status[filename] = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    return model_status


def ensure_model_directories():
    try:
        import os
        
        required_dirs = [
            os.path.join(folder_paths.models_dir, "forbidden_vision"),
            os.path.join(folder_paths.models_dir, "upscale_models"),
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
        return True
        
    except Exception as e:
        print(f"Error creating model directories: {e}")
        return False

def get_ordered_upscaler_model_list():
    fast_options = [
        "Fast 4x (Bicubic AA)",
        "Fast 4x (Lanczos)",
        "Fast 2x (Bicubic AA)",
        "Fast 2x (Lanczos)"
    ]
    
    all_models = folder_paths.get_filename_list("upscale_models") or []
    final_list = fast_options.copy()
    final_list.extend(sorted(all_models))
    
    return final_list



try:
    from .depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False

class DepthAnythingManager:
    _instance = None
    _model_cache = {}
    _transform_cache = {}
    
    def __init__(self):
        if not DEPTH_ANYTHING_AVAILABLE:
            print("WARNING: Depth Anything V2 module not available")
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _download_with_progress(self, url, file_path, description=""):
        class ProgressTracker:
            def __init__(self, description):
                self.pbar = None
                self.description = description
                self.last_percent = -1

            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar_total = total_size if total_size != -1 else 1000000000
                    self.pbar_unit = 'B' if total_size != -1 else 'iB'
                    self.pbar_unit_scale = True
                    self.pbar_unit_divisor = 1024
                    
                    print(f"Depth Manager: Downloading {self.description}")
                    self.pbar = True

                downloaded = block_num * block_size
                percent = int(downloaded * 100 / total_size) if total_size != -1 else -1

                if percent > self.last_percent and percent < 100:
                    self.last_percent = percent
                    bar_length = 40
                    filled_len = int(bar_length * downloaded // total_size)
                    bar = '#' * filled_len + '-' * (bar_length - filled_len)
                    
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    
                    print(f"  [{bar}] {percent:.0f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='\r')

        try:
            import urllib.request
            tracker = ProgressTracker(description)
            urllib.request.urlretrieve(url, file_path, reporthook=tracker)
            print("\nDepth Manager: Download complete.")
        except Exception as e:
            print(f"\nDepth Manager: Download failed: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    def load_depth_model(self, model_name="V2-Small"):
        check_for_interruption()
        
        if model_name in self._model_cache and model_name in self._transform_cache:
            return self._model_cache[model_name], self._transform_cache[model_name]
        
        MODEL_CONFIGS = {
            "V2-Small": {
                "config": {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                "filename": "depth_anything_v2_vits.pth",
                "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
            },
            "V2-Base": {
                "config": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                "filename": "depth_anything_v2_vitb.pth",
                "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
            }
        }

        try:
            device = model_management.get_torch_device()
            
            if not DEPTH_ANYTHING_AVAILABLE:
                print("Depth Manager: Depth Anything V2 module not available")
                return None, None

            selected_model = MODEL_CONFIGS.get(model_name)
            if not selected_model:
                print(f"Depth Manager: Unknown depth model '{model_name}'. Falling back to V2-Small.")
                selected_model = MODEL_CONFIGS["V2-Small"]
                model_name = "V2-Small"

            model_config = selected_model["config"]
            filename = selected_model["filename"]
            url = selected_model["url"]
            
            model = DepthAnythingV2(**model_config)
            
            model_dir = os.path.join(folder_paths.models_dir, "depth_anything_v2")
            model_path = os.path.join(model_dir, filename)
            
            if not os.path.exists(model_path):
                os.makedirs(model_dir, exist_ok=True)
                self._download_with_progress(url, model_path, f"Depth Anything {model_name}")
            
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            if device.type == "cuda":
                model = model.half()

            def transform_and_predict(img_np_rgb):
                try:
                    check_for_interruption()

                    if img_np_rgb.dtype != np.uint8:
                        img_np_rgb = (img_np_rgb * 255).astype(np.uint8)

                    img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

                    dev = next(model.parameters()).device
                    use_amp = (dev.type == "cuda")
                    amp_dtype = torch.float16

                    with torch.inference_mode():
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                depth = model.infer_image(img_bgr)
                        else:
                            depth = model.infer_image(img_bgr)

                    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    return depth_normalized

                except model_management.InterruptProcessingException:
                    raise
                except Exception as e:
                    print(f"Error in Depth Anything prediction: {e}")
                    return np.zeros_like(img_np_rgb[:, :, 0], dtype=np.float32)

            
            self._model_cache[model_name] = model
            self._transform_cache[model_name] = transform_and_predict
            
            print(f"Depth Manager: Successfully loaded {model_name}")
            return model, transform_and_predict
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Depth Manager: Failed to load {model_name}: {e}")
            return None, None
    
    def infer_depth_cropped(self, image_crop_rgb, model_name="V2-Small", crop_coords=None):
        try:
            check_for_interruption()
            
            depth_model, depth_transform = self.load_depth_model(model_name)
            if depth_model is None or depth_transform is None:
                print(f"Depth Manager: Model {model_name} not available for cropped inference")
                return None, crop_coords
            
            if image_crop_rgb is None or image_crop_rgb.size == 0:
                print("Depth Manager: Invalid crop image provided")
                return None, crop_coords
            
            crop_h, crop_w = image_crop_rgb.shape[:2]
            if crop_h < 32 or crop_w < 32:
                print(f"Depth Manager: Crop too small ({crop_w}x{crop_h}), minimum 32x32")
                return None, crop_coords
            
            if image_crop_rgb.dtype != np.uint8:
                image_crop_rgb = (image_crop_rgb * 255).astype(np.uint8)
            
            depth_crop_np = depth_transform(image_crop_rgb)
            
            if depth_crop_np is None or depth_crop_np.size == 0:
                print("Depth Manager: Depth inference failed on crop")
                return None, crop_coords
            
            depth_crop_tensor = torch.from_numpy(depth_crop_np).unsqueeze(0).unsqueeze(0).float()
            
            min_val = torch.min(depth_crop_tensor)
            max_val = torch.max(depth_crop_tensor)
            if max_val > min_val:
                depth_crop_normalized = (depth_crop_tensor - min_val) / (max_val - min_val)
            else:
                depth_crop_normalized = torch.zeros_like(depth_crop_tensor)
            
            return depth_crop_normalized, crop_coords
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Depth Manager: Error in cropped inference: {e}")
            return None, crop_coords

    def infer_depth_full(self, image_rgb, model_name="V2-Small"):
        try:
            check_for_interruption()
            
            depth_model, depth_transform = self.load_depth_model(model_name)
            if depth_model is None or depth_transform is None:
                return None
            
            h, w = image_rgb.shape[:2]
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            depth_np = depth_transform(image_rgb)
            
            if depth_np is None:
                return None
            
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
            depth_tensor = depth_tensor.float()
            
            min_val = torch.min(depth_tensor)
            max_val = torch.max(depth_tensor)
            if max_val > min_val:
                depth_normalized = (depth_tensor - min_val) / (max_val - min_val)
            else:
                depth_normalized = torch.zeros_like(depth_tensor)
            
            return depth_normalized
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Depth Manager: Error in full image inference: {e}")
            return None

    def clear_cache(self):
        self._model_cache.clear()
        self._transform_cache.clear()
        print("Depth Manager: Cache cleared")