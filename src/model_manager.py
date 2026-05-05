import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import folder_paths
import kornia
import comfy.model_management as model_management
from .utils import check_for_interruption
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .neural_train import (
    rgb_to_yuv_bt601,
    yuv_to_rgb_bt601,
    BilateralGridEditor,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Neural corrector will be unavailable.")


class ForbiddenVisionModelManager:
    _instance = None

    MODELS_CONFIG = {
        'face_detect': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models',
            'filename': 'ForbiddenVision_face_detect_v1.pt',
            'model_type': 'yolo'
        },
        'face_detect_onnx': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models',
            'filename': 'ForbiddenVision_face_detect_v1.onnx',
            'model_type': 'yolo'
        },
        'face_segment': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models',
            'filename': 'ForbiddenVision_face_segment_v1.safetensors',
            'model_type': 'unetplusplus'
        },
        'neural_corrector': {
            'repo_id': 'luxdelux7/ForbiddenVision_Models',
            'filename': 'ForbiddenVision_neural_corrector_v1.safetensors',
            'model_type': 'rgb_curves_v2'
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
        self._models = {}

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
                        try: os.remove(local_path)
                        except: pass
            return None

        except Exception as e:
            print(f"ForbiddenVision: Critical error downloading {model_key}: {e}")
            return None

    def validate_model_availability(self):
        status = {
            'face_detection': False,
            'face_segmentation': False,
            'neural_corrector': False
        }

        for key in ['face_detect', 'face_segment', 'neural_corrector']:
            config = self.MODELS_CONFIG[key]
            local_path = os.path.join(self.models_dir, config['filename'])
            if key == 'face_detect': status_key = 'face_detection'
            elif key == 'face_segment': status_key = 'face_segmentation'
            else: status_key = key
            status[status_key] = os.path.exists(local_path) and os.path.getsize(local_path) > 1000

        return status

    def initialize_default_models(self):
        print("ForbiddenVision: Checking default models...")

        required_models = {
            'face_detect': 'Face Detection',
            'face_segment': 'Face Segmentation',
            'neural_corrector': 'Neural Color/Tone Corrector'
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

        print(f"ForbiddenVision: {successful_downloads}/{len(required_models)} models ready")
        return download_results

    def _check_onnx_available(self):
        """Check if onnxruntime is available for ONNX model loading."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def load_face_detection_model(self):
        model_name_pt   = 'ForbiddenVision_face_detect_v1.pt'
        model_name_onnx = 'ForbiddenVision_face_detect_v1.onnx'

        device = model_management.get_torch_device()

        for name in (model_name_onnx, model_name_pt):
            if name in self._models:
                cached = self._models[name]
                # Skip device transfer for ONNX — onnxruntime handles its own placement
                if name.endswith('.onnx'):
                    return cached
                # Move PyTorch model back to GPU if it was offloaded
                target = cached.model if hasattr(cached, 'model') and hasattr(cached.model, 'to') else cached
                try:
                    params = list(target.parameters()) if hasattr(target, 'parameters') else []
                    if params and params[0].device != device:
                        target.to(device)
                except Exception:
                    pass
                return cached

        onnx_available = self._check_onnx_available()
        is_cuda = device.type == "cuda"

        use_onnx = False
        if onnx_available:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if is_cuda and "CUDAExecutionProvider" in providers:
                    use_onnx = True
                    print("ForbiddenVision: ONNX runtime with CUDA support detected")
                elif is_cuda and "ROCMExecutionProvider" in providers:
                    use_onnx = True
                    print("ForbiddenVision: ONNX runtime with ROCm support detected")
                elif not is_cuda:
                    use_onnx = True
                    print("ForbiddenVision: ONNX runtime (CPU) detected")
                else:
                    print("ForbiddenVision: onnxruntime installed but no GPU provider, falling back to .pt")
            except Exception:
                pass

        if use_onnx:
            local_path = os.path.join(self.models_dir, model_name_onnx)
            if not os.path.exists(local_path):
                self._download_model('face_detect_onnx')
            if not os.path.exists(local_path):
                use_onnx = False

        # Try loading ONNX with a smoke test
        if use_onnx:
            local_path_onnx = os.path.join(self.models_dir, model_name_onnx)
            try:
                model = YOLO(local_path_onnx, task='detect')
                # Smoke test: run a tiny dummy inference to catch runtime errors early
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                device_str = "0" if is_cuda else "cpu"
                model.predict(dummy, conf=0.9, verbose=False, device=device_str)
                self._models[model_name_onnx] = model
                print("ForbiddenVision: Loaded face detection model [ONNX]")
                return model
            except Exception as e:
                print(f"ForbiddenVision: ONNX runtime failed smoke test ({e}), falling back to PyTorch")
                self._models.pop(model_name_onnx, None)
                use_onnx = False

        # PyTorch fallback
        local_path_pt = os.path.join(self.models_dir, model_name_pt)
        if not os.path.exists(local_path_pt):
            self._download_model('face_detect')

        if not os.path.exists(local_path_pt):
            return None

        try:
            model = YOLO(local_path_pt, task='detect')
            model.to(device)
            self._models[model_name_pt] = model
            print("ForbiddenVision: Loaded face detection model [PyTorch]")
            return model
        except Exception as e:
            print(f"ForbiddenVision: Error loading face detection model: {e}")
            return None

    def load_segmentation_model(self):
        check_for_interruption()
        if self.segmentation_model is not None:
            device = model_management.get_torch_device()
            try:
                params = list(self.segmentation_model.parameters())
                if params and params[0].device != device:
                    self.segmentation_model.to(device)
            except Exception:
                pass
            return self.segmentation_model

        try:
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.encoders import get_preprocessing_fn

            config = self.MODELS_CONFIG['face_segment']
            model_path = os.path.join(self.models_dir, config['filename'])

            if not os.path.exists(model_path):
                model_path = self._download_model('face_segment')
                if not model_path: return None

            device = model_management.get_torch_device()
            model = smp.UnetPlusPlus(
                encoder_name="tu-tf_efficientnetv2_s.in21k_ft_in1k",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                decoder_channels=(256, 128, 64, 32, 16)
            )

            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(model_path, device='cpu')
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.half().to(device).eval()

            self.segmentation_model = model
            self.segmentation_preprocessing = get_preprocessing_fn(
                "tu-tf_efficientnetv2_s.in21k_ft_in1k",
                pretrained="imagenet"
            )
            print("Successfully loaded face segmentation model")
            return model
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            return None

    def load_neural_corrector(self):
        check_for_interruption()
        if 'neural_corrector' in self._models:
            cached = self._models['neural_corrector']
            device = model_management.get_torch_device()
            try:
                params = list(cached.parameters())
                if params and params[0].device != device:
                    cached.to(device)
            except Exception:
                pass
            return cached

        if not TIMM_AVAILABLE:
            print("ForbiddenVision: timm not installed, neural corrector unavailable")
            return None

        try:
            config = self.MODELS_CONFIG['neural_corrector']
            local_path = os.path.join(self.models_dir, config['filename'])

            if not os.path.exists(local_path):
                local_path = self._download_model('neural_corrector')

            if not local_path or not os.path.exists(local_path):
                print("ForbiddenVision: Failed to acquire Corrector model.")
                return None

            device = model_management.get_torch_device()

            model = BilateralGridEditor(
                backbone_name='mobilenetv4_conv_small.e2400_r224_in1k',
                grid_d=24,
                pretrained=False
            )

            if local_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(local_path, device="cpu")
            else:
                try:
                    ckpt = torch.load(local_path, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(local_path, map_location="cpu")

                if isinstance(ckpt, dict):
                    if "model" in ckpt and isinstance(ckpt["model"], dict):
                        state_dict = ckpt["model"]
                    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                        state_dict = ckpt["state_dict"]
                    else:
                        state_dict = ckpt
                else:
                    raise RuntimeError("Checkpoint is not a dict / state_dict")

            model.load_state_dict(state_dict, strict=True)

            model = model.to(device)
            if device.type == "cuda":
                model = model.half()
                dtype_label = "FP16"
            else:
                model = model.float()
                dtype_label = "FP32"

            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            self._models['neural_corrector'] = model
            print(f"ForbiddenVision: Loaded BilateralGridEditor Neural Corrector [{dtype_label}]")
            return model

        except Exception as e:
            print(f"ForbiddenVision: Failed to load Corrector: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_neural_corrector(
        self,
        image_bchw: torch.Tensor,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
    ) -> tuple:
        """Run neural corrector single pass with tone/color strength blend."""
        check_for_interruption()
        model = self.load_neural_corrector()
        if model is None:
            return image_bchw.float(), {}

        model_param = next(model.parameters())
        model_device = model_param.device
        model_dtype = model_param.dtype

        x = image_bchw.to(device=model_device, dtype=model_dtype).clamp(0.0, 1.0)

        with torch.no_grad():
            corrected, aux = _forward_with_strength(
                model, x, tone_strength=1.0, color_strength=1.0
            )
            corrected = corrected.clamp(0.0, 1.0)

            orig = x.float()
            final = corrected.float()

            def luma(img):
                return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

            def clip_color(img):
                L = luma(img)
                mn = img.min(dim=1, keepdim=True).values
                mx = img.max(dim=1, keepdim=True).values
                img = torch.where(mn < 0.0, L + (img - L) * (L / (L - mn + 1e-6)), img)
                mx = img.max(dim=1, keepdim=True).values
                img = torch.where(mx > 1.0, L + (img - L) * ((1.0 - L) / (mx - L + 1e-6)), img)
                return img

            def set_lum(img, L_target):
                return clip_color(img + (L_target - luma(img)))

            L_orig = luma(orig)
            L_final = luma(final)

            base = torch.lerp(orig, set_lum(final, L_orig), float(color_strength)) if color_strength > 0 else orig
            out = torch.lerp(base, set_lum(base, L_final), float(tone_strength)) if tone_strength > 0 else base

            return out.clamp(0.0, 1.0).float(), aux

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

    def offload_to_cpu(self):
        """Move all loaded models from VRAM to system RAM to free GPU memory.
        Models stay in self._models / self.segmentation_model and will be
        moved back to GPU on next use.
        
        Note: ONNX models are skipped — they're managed by onnxruntime, not
        PyTorch, and their device is fixed at session creation."""
        try:
            for key, m in self._models.items():
                # Skip ONNX YOLO models — onnxruntime manages its own memory
                if key.endswith('.onnx'):
                    continue
                
                # YOLO wraps the actual nn.Module in .model
                target = m.model if hasattr(m, 'model') and hasattr(m.model, 'to') else m
                if hasattr(target, 'to'):
                    try:
                        target.to('cpu')
                    except Exception as e:
                        print(f"ForbiddenVision: Could not offload {key}: {e}")

            if self.segmentation_model is not None:
                try:
                    self.segmentation_model.to('cpu')
                except Exception as e:
                    print(f"ForbiddenVision: Could not offload segmentation model: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"ForbiddenVision: Error during CPU offload: {e}")


@torch.no_grad()
def shadow_crush_restore(
    base: torch.Tensor,
    original: torch.Tensor,
    opacity: float = 0.15,
) -> torch.Tensor:
    low  = base - (1.0 - 2.0 * original) * base * (1.0 - base)
    high = base + (2.0 * original - 1.0) * (torch.sqrt(base.clamp(min=1e-8)) - base)
    soft = torch.where(original <= 0.5, low, high)
    crush_only = torch.min(soft, base)
    return (crush_only * opacity + base * (1.0 - opacity)).clamp(0.0, 1.0)

@torch.no_grad()
def _forward_with_strength(
    model: BilateralGridEditor,
    x: torch.Tensor,
    tone_strength: float = 1.0,
    color_strength: float = 1.0,
) -> tuple:
    x = x.clamp(0.0, 1.0)
    B, C, H, W = x.shape
    target_size = 384

    if W >= H:
        new_w = target_size
        new_h = max(8, int(round(H * (target_size / W))))
    else:
        new_h = target_size
        new_w = max(8, int(round(W * (target_size / H))))

    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)

    x_small = F.interpolate(
        x, size=(new_h, new_w),
        mode='bilinear', align_corners=False, antialias=True
    )

    model_dtype = next(model.parameters()).dtype
    model.float()
    x_small_f32 = x_small.float()
    x_f32 = x.float()

    model.eval()
    pred_small, aux = model(x_small_f32, x_small_f32)

    params = {
        "ev": aux["ev"],
        "hi": aux["hi"],
        "curve": aux["curve"],
        "a_grid": aux["a_grid"],
        "b_grid": aux["b_grid"],
        "temp": aux["temp"],
        "tint": aux["tint"],
        "shadows": aux["shadows"],
        "chroma_grid": aux["chroma_grid"],
    }

    if max(H, W) > 900:
        pred = model.apply_params_chunked(x_f32, params, chunk_h=512, chunk_w=512)
    else:
        pred = model.apply_params(x_f32, params)

    if model_dtype == torch.float16:
        model.half()

    pred = pred.clamp(0.0, 1.0)

    t = float(tone_strength)
    c = float(color_strength)

    if t >= 0.999 and c >= 0.999:
        return pred.float(), aux

    orig = x.float()
    final = pred.float()

    def luma(img):
        return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

    def clip_color(img):
        L = luma(img)
        mn = img.min(dim=1, keepdim=True).values
        mx = img.max(dim=1, keepdim=True).values
        img = torch.where(mn < 0.0, L + (img - L) * (L / (L - mn + 1e-6)), img)
        mx = img.max(dim=1, keepdim=True).values
        img = torch.where(mx > 1.0, L + (img - L) * ((1.0 - L) / (mx - L + 1e-6)), img)
        return img

    def set_lum(img, L_target):
        return clip_color(img + (L_target - luma(img)))

    L_orig = luma(orig)
    L_final = luma(final)

    base = torch.lerp(orig, set_lum(final, L_orig), c) if c > 0 else orig
    out = torch.lerp(base, set_lum(base, L_final), t) if t > 0 else base
    out = shadow_crush_restore(out, orig, opacity=0.10)
    return out.clamp(0.0, 1.0).float(), aux

