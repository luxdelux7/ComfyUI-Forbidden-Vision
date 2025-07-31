import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import nodes
import kornia
import os
import sys
import cv2
import urllib.request
from PIL import Image
from . import mood_presets
from .utils import check_for_interruption, get_ordered_upscaler_model_list

try:
    from transparent_background import Remover
except ImportError:
    Remover = None

try:
    from skimage import exposure
    from skimage.color import rgb2lab, lab2rgb
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False

if Remover is None:
    print("-------------------------------------------------------------------------------------------------")
    print("WARNING: transparent-background library not found.")
    print("Please install it running requirements.txt")
    print("-------------------------------------------------------------------------------------------------")   

class LatentRefiner:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()
        
        simple_method = "Simple: Bicubic (Standard)"
        upscaler_models.insert(0, simple_method)

        return {
            "required": {

                "enable_upscale": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "upscale_model": (upscaler_models, ),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.05}),
                
                "enable_auto_tone": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "label": "Enable Auto Tone Correction"}),

                "enable_dof_effect": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "dof_blur_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                
                "relighting_mode": (["Disabled", "Additive (Simple)", "Corrective"],),
                "relight_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mood_preset": (list(mood_presets.PRESETS.keys()), ), 
                "mood_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),

                
                "enable_vibrance": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "vibrance_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                "enable_clarity": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "clarity_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.5, "step": 0.05}),
                
                "enable_smart_sharpen": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "sharpening_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                
            },
            "optional": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mood_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "refined_image_preview")
    FUNCTION = "refine_and_process"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        self.upscaler_model = None
        self.upscaler_model_name = None
        self.remover = None
        self.depth_model = None
        self.depth_transform = None
        
        self.cached_input_tensor = None
        self.cached_subject_mask = None
        self.cached_depth_map = None
   
    def _run_and_cache_analysis(self, image_tensor, run_segmentation, run_depth):
     
        try:
            check_for_interruption()
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            device = image_tensor.device
            
            self.cached_subject_mask = None
            self.cached_depth_map = None
            
            img_np_uint8 = None
            
            if run_segmentation:

                if self.remover is None:
                    if Remover is not None:
                        try:
                            self.remover = Remover(mode='base', jit=False)
                        except Exception as e:
                            print(f"Failed to initialize transparent-background remover: {e}. Segmentation skipped.")
                            run_segmentation = False
                    else:
                        print("transparent-background library not available. Segmentation skipped.")
                        run_segmentation = False

                if run_segmentation:
                    img_np_uint8 = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_np_uint8)
                    mask_pil = self.remover.process(pil_image, type='map').convert('L')
                    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                    self.cached_subject_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

            check_for_interruption()

            if run_depth:
            
                depth_model, depth_transform = self.load_depth_model()
                if depth_model and depth_transform:
                    if img_np_uint8 is None:
                        img_np_uint8 = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                    
                    try:
                        depth_np = depth_transform(img_np_uint8)
                        
                        if isinstance(depth_np, np.ndarray) and depth_np.ndim == 2:
                            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device).float()
                            
                            depth_tensor_resized = F.interpolate(
                                depth_tensor, size=(h, w), mode="bilinear", align_corners=False
                            )
                            
                            min_val = torch.min(depth_tensor_resized)
                            max_val = torch.max(depth_tensor_resized)
                            if max_val > min_val:
                                self.cached_depth_map = (depth_tensor_resized - min_val) / (max_val - min_val)
                            else:
                                self.cached_depth_map = torch.zeros_like(depth_tensor_resized)

                        else:
                            print(f"Warning: Unexpected depth output format: {type(depth_np)}, shape: {depth_np.shape if hasattr(depth_np, 'shape') else 'unknown'}")
                            
                    except Exception as e:
                        print(f"Warning: Depth estimation failed: {e}")
 
                else:
                    print("Refiner: Depth model not available, skipping depth estimation.")
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: An error occurred during AI analysis: {e}")
            self.cached_input_tensor = None

    def load_depth_model(self):

        model_name_to_check = "depth_anything_v2_vits"
        
        if self.depth_model is not None and self.depth_transform is not None:
            if hasattr(self.depth_model, "name_of_model_for_refiner") and self.depth_model.name_of_model_for_refiner == model_name_to_check:
                return self.depth_model, self.depth_transform
            else:
                self.depth_model = None
                self.depth_transform = None
                print(f"Refiner: Switching to {model_name_to_check}.")
        
        try:
            device = model_management.get_torch_device()
            
            if not DEPTH_ANYTHING_AVAILABLE:
                print("FATAL: Depth Anything V2 module not available")
                self.depth_model = None
                self.depth_transform = None
                return None, None

            model_config = {
                'encoder': 'vits', 
                'features': 64, 
                'out_channels': [48, 96, 192, 384]
            }
            
            model = DepthAnythingV2(**model_config)
            
            model_dir = os.path.join(folder_paths.models_dir, "depth_anything_v2")
            model_path = os.path.join(model_dir, "depth_anything_v2_vits.pth")
            
            if not os.path.exists(model_path):
                print(f"Refiner: Downloading Depth Anything V2 Small weights to '{model_path}'...")
                os.makedirs(model_dir, exist_ok=True)
                url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
                urllib.request.urlretrieve(url, model_path)
                print(f"Refiner: Download complete.")
            
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            
            def transform_and_predict(img_np_rgb):
                try:
                    check_for_interruption()
                    
                    if img_np_rgb.dtype != np.uint8:
                        img_np_rgb = (img_np_rgb * 255).astype(np.uint8)
                    
                    img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
                    
                    depth = model.infer_image(img_bgr)
                    
                    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
                    
                    return depth_normalized
                    
                except model_management.InterruptProcessingException:
                    raise
                except Exception as e:
                    print(f"Error in Depth Anything V2 prediction: {e}")
                    return np.zeros_like(img_np_rgb[:,:,0])
            
            self.depth_model = model
            self.depth_model.name_of_model_for_refiner = model_name_to_check
            self.depth_transform = transform_and_predict
            
            return self.depth_model, self.depth_transform
            
        except Exception as e:
            print(f"FATAL: Failed to load Depth Anything V2: {e}")
            print("Refiner: Falling back to disabled depth estimation.")

            self.depth_model = None
            self.depth_transform = None
            return None, None
        
    def load_upscaler_model(self, model_name):
        if self.upscaler_model is not None and self.upscaler_model_name == model_name:
            return self.upscaler_model
        try:
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            self.upscaler_model = upscaler_loader.load_model(model_name)[0]
            self.upscaler_model_name = model_name
            return self.upscaler_model
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error loading upscaler model {model_name}: {e}")
            self.upscaler_model = None; self.upscaler_model_name = None
            return None
    
    def analyze_image_to_preset(self, mood_image, target_h, target_w):
       
        try:
            check_for_interruption()
            device = mood_image.device
            
            mood_image_bchw = mood_image.permute(0, 3, 1, 2)
            orig_h, orig_w = mood_image_bchw.shape[-2:]
            orig_aspect, target_aspect = orig_w / orig_h, target_w / target_h
            new_h, new_w = (int(target_w / orig_aspect), target_w) if orig_aspect > target_aspect else (target_h, int(target_h * orig_aspect))
            
            resized_mood_image = F.interpolate(mood_image_bchw, size=(new_h, new_w), mode='bilinear', align_corners=False)

            image_lab = kornia.color.rgb_to_lab(resized_mood_image)
            l_channel, a_channel, b_channel = image_lab[:, 0:1], image_lab[:, 1:2], image_lab[:, 2:3]
            l_normalized = l_channel / 100.0
            saturation = torch.sqrt(a_channel**2 + b_channel**2)
            saturation_normalized = saturation / (saturation.max() + 1e-6)

            numeric_brightness = torch.mean(l_normalized).item()
            numeric_contrast = torch.std(l_normalized).item()
            numeric_temperature = torch.mean(b_channel).item()
            chromatic_intensity_map = l_normalized * saturation_normalized
            avg_emissive_intensity = torch.mean(torch.topk(chromatic_intensity_map.view(-1), int(chromatic_intensity_map.numel() * 0.005)).values).item()
            
            emissive_mask = (chromatic_intensity_map > torch.quantile(chromatic_intensity_map, 0.995)).float()
            
            emissive_pixels_saturation = saturation_normalized[emissive_mask.bool()]

            avg_emissive_saturation = torch.mean(emissive_pixels_saturation).item() if emissive_pixels_saturation.numel() > 0 else 0
            black_point_presence = torch.mean((l_normalized < 0.1).float()).item()
            
            emissive_colors_raw = self._extract_colors_with_kmeans(resized_mood_image, emissive_mask, 2, device)
            highlight_mask = (l_normalized > 0.85).float()
            shadow_mask = (l_normalized < 0.15).float()
            highlight_tint = resized_mood_image[highlight_mask.bool().expand_as(resized_mood_image)].view(3, -1).mean(dim=1).tolist() if highlight_mask.sum() > 0 else [1,1,1]
            shadow_tint = resized_mood_image[shadow_mask.bool().expand_as(resized_mood_image)].view(3, -1).mean(dim=1).tolist() if shadow_mask.sum() > 0 else [0,0,0]
            midtone_mask = (l_normalized >= 0.15) & (l_normalized <= 0.85)
            accent_colors = self._extract_colors_with_kmeans(resized_mood_image, midtone_mask.float(), 3, device)
            

            darkness_score = 1.0 - numeric_brightness
            contrast_score = min(numeric_contrast / 0.35, 1.0)
            warmth_score = max(0, min(numeric_temperature / 20.0, 1.0))
            coolness_score = max(0, min(-numeric_temperature / 20.0, 1.0))
            emissive_score = min(avg_emissive_intensity / 0.7, 1.0)

            mood_scores = {}

            mood_scores["bright_natural"] = (
                warmth_score * 0.3 +
                (1.0 - darkness_score) * 0.25 +
                (1.0 - black_point_presence) * 0.2 +
                min(contrast_score / 0.6, 1.0) * 0.15 +
                (1.0 - emissive_score * 0.5) * 0.1
            )

            mood_scores["warm_cinematic"] = (
                warmth_score * 0.25 +
                contrast_score * 0.25 +
                min(black_point_presence * 5.0, 1.0) * 0.2 +
                min(darkness_score * 1.5, 1.0) * 0.15 +
                min(emissive_score * 1.2, 1.0) * 0.15
            )

            mood_scores["cool_dramatic"] = (
                coolness_score * 0.3 +
                darkness_score * 0.25 +
                contrast_score * 0.2 +
                min(black_point_presence * 4.0, 1.0) * 0.15 +
                (1.0 - warmth_score) * 0.1
            )

            mood_scores["cyberpunk_vibrant"] = (
                emissive_score * 0.35 +
                min(avg_emissive_saturation * 1.2, 1.0) * 0.25 +
                contrast_score * 0.2 +
                (1.0 - warmth_score * 0.7) * 0.1 +
                min((coolness_score + 0.3), 1.0) * 0.1
            )

            mood_type = max(mood_scores.keys(), key=lambda k: mood_scores[k])

            temperature_adjustment = numeric_temperature
            brightness_adjustment = (numeric_brightness - 0.7) * -40.0
            brightness_adjustment = min(0, brightness_adjustment)
            
            if mood_type == "cyberpunk_vibrant":
                atmospheric_effects = {"type": "cyberpunk_complex", "glow_colors": emissive_colors_raw, "complex_glows": True,
                                        "intensity_multiplier": 1.8, "saturation_boost": 1.7, "coverage_multiplier": 1.3}
            elif mood_type == "cool_dramatic":
                atmospheric_effects = {"type": "mysterious_cool", "glow_colors": [highlight_tint], 
                                        "intensity_multiplier": 1.4, "saturation_boost": 1.2, "coverage_multiplier": 1.1}
            elif mood_type == "warm_cinematic":
                atmospheric_effects = {"type": "warm_cinematic", "glow_colors": [highlight_tint],
                                        "intensity_multiplier": 1.5, "saturation_boost": 1.3, "coverage_multiplier": 1.2}
            else:
                atmospheric_effects = {"type": "clean_natural", "glow_colors": [highlight_tint],
                                        "intensity_multiplier": 1.2, "saturation_boost": 1.1, "coverage_multiplier": 1.0}

            generated_preset = {
                "mood_type": mood_type,
                "color_palette": { "highlight_tint": highlight_tint, "shadow_tint": shadow_tint, "accent_colors": accent_colors },
                "temperature_shift": temperature_adjustment,
                "brightness_shift": brightness_adjustment,
                "atmospheric_effects": atmospheric_effects
            }
            
            return generated_preset

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: Error analyzing mood image: {e}")

            return {}
    def refine_and_process(self,
                        enable_auto_tone,
                        enable_upscale, upscale_model, upscale_factor,
                        enable_dof_effect, dof_blur_strength,
                        relighting_mode, relight_strength,
                        mood_preset, mood_strength,
                        enable_vibrance, vibrance_strength,
                        enable_clarity, clarity_strength,
                        enable_smart_sharpen, sharpening_strength,
                        use_tiled_vae, tile_size,
                        latent=None, vae=None, image=None, mood_image=None, **kwargs):
        try:
            check_for_interruption()
            device = model_management.get_torch_device()

            is_latent_input = latent is not None and "samples" in latent
            is_image_input = image is not None
            
            if not is_latent_input and not is_image_input:
                print("Warning: No valid inputs provided. Please connect either (latent + vae) or image.")
                return ({"samples": torch.zeros((1, 4, 64, 64))}, torch.zeros((1, 64, 64, 3)))
            
            input_key_tensor = latent["samples"] if is_latent_input else image
            
            decoded_image = None
            if is_image_input:
                decoded_image = image.to(device)
            elif is_latent_input and vae is not None:
                decoded_image = vae.decode(input_key_tensor.to(device))
            
            if decoded_image is None:
                print("Warning: Cannot decode image. Processing cannot continue without a VAE for latents.")
                dummy_latent = latent if is_latent_input else {"samples": torch.zeros((1, 4, 64, 64))}
                dummy_image = image if is_image_input else torch.zeros((1, 64, 64, 3))
                return (dummy_latent, dummy_image)

            image_to_process = decoded_image
            
            image_to_process = self.prepare_source_dynamics(image_to_process)

            image_bchw = image_to_process.permute(0, 3, 1, 2)
            
            is_mood_preset_active = (mood_preset != "Disabled" or mood_image is not None) and mood_strength > 0
            is_relighting_active = relighting_mode != "Disabled" and relight_strength > 0
            is_dof_active = enable_dof_effect and dof_blur_strength > 0
            is_segmentation_needed = is_mood_preset_active or is_relighting_active or is_dof_active
            is_depth_needed = is_dof_active

            is_cache_valid = (
                self.cached_input_tensor is not None and
                self.cached_input_tensor.shape == input_key_tensor.shape and
                torch.equal(self.cached_input_tensor, input_key_tensor) and
                (not is_segmentation_needed or self.cached_subject_mask is not None) and
                (not is_depth_needed or self.cached_depth_map is not None)
            )
            
            if not is_cache_valid and (is_segmentation_needed or is_depth_needed):
                self.cached_input_tensor = input_key_tensor.clone()
                self._run_and_cache_analysis(decoded_image, is_segmentation_needed, is_depth_needed)
            
            if is_mood_preset_active:
                preset_data = {}
                if mood_image is not None:
                    target_h, target_w = image_bchw.shape[-2:]
                    preset_data = self.analyze_image_to_preset(mood_image, target_h, target_w)
                else:
                    preset_data = mood_presets.PRESETS.get(mood_preset, {})
                
                if preset_data:
                    image_bchw = self.apply_mood_and_lighting_transfer(
                        image_bchw, self.cached_subject_mask, self.cached_depth_map, preset_data, mood_strength
                    )

            final_foreground = image_bchw.clone()
            final_background = image_bchw.clone()

            if is_dof_active and self.cached_depth_map is not None and self.cached_subject_mask is not None:
                final_background = self._apply_dof_pyramid(final_background, self.cached_subject_mask, self.cached_depth_map, dof_blur_strength, device)

            if is_relighting_active and self.cached_subject_mask is not None:
                if self.cached_subject_mask.sum() > 0:
                    adjusted_relight_strength = relight_strength * 0.5 if is_mood_preset_active else relight_strength
                    if relighting_mode == "Additive (Simple)":
                        final_foreground = self.apply_professional_relighting(final_foreground, self.cached_subject_mask, (adjusted_relight_strength ** 0.7 * 1.8), device)
                    elif relighting_mode == "Corrective":
                        final_foreground = self.apply_correction_photo(final_foreground, self.cached_subject_mask, adjusted_relight_strength)

            if self.cached_subject_mask is not None and (is_dof_active or is_relighting_active):
                compositing_mask = self.cached_subject_mask
                target_h, target_w = final_foreground.shape[-2:]
                if final_background.shape[-2:] != (target_h, target_w) or compositing_mask.shape[-2:] != (target_h, target_w):
                    final_background = F.interpolate(final_background, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    compositing_mask = F.interpolate(compositing_mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
                final_bchw = torch.lerp(final_background, final_foreground, compositing_mask)
            else:
                final_bchw = final_foreground
            
            final_bhwc = final_bchw.permute(0, 2, 3, 1)

            if enable_vibrance: final_bhwc = self.apply_vibrance_gpu(final_bhwc, vibrance_strength)
            if enable_clarity: final_bhwc = self.apply_clarity_gpu(final_bhwc, clarity_strength)

            if enable_auto_tone:
                final_bhwc = self.finalize_tone_correction(final_bhwc)

            final_image = final_bhwc

            if enable_upscale and upscale_factor > 1.0:
                h_orig, w_orig = final_image.shape[1], final_image.shape[2]
                target_h, target_w = int(h_orig * upscale_factor), int(w_orig * upscale_factor)
                if upscale_model == "Simple: Bicubic (Standard)":
                    final_image = F.interpolate(final_image.movedim(-1, 1), size=(target_h, target_w), mode='bicubic', align_corners=False, antialias=True).movedim(1, -1)
                else:
                    loaded_model = self.load_upscaler_model(upscale_model)
                    if loaded_model:
                        ai_upscaled_image = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']().upscale(upscale_model=loaded_model, image=final_image)[0]
                        final_image = F.interpolate(ai_upscaled_image.movedim(-1, 1), size=(target_h, target_w), mode='bicubic', align_corners=False, antialias=True).movedim(1, -1)
                    else:
                        print(f"Warning: Upscaler model {upscale_model} failed to load. Skipping upscale.")

            if enable_smart_sharpen and sharpening_strength > 0:
                final_image = self.apply_smart_sharpen(final_image, sharpening_strength)

            final_image = final_image.to(device)
            
            refined_latent = None
            if vae is not None:
                clamped_final_image = torch.clamp(final_image, 0.0, 1.0)
                if use_tiled_vae:
                    encode_node = nodes.NODE_CLASS_MAPPINGS['VAEEncodeTiled']()
                    refined_latent = encode_node.encode(vae, clamped_final_image, tile_size)[0]
                else:
                    encode_node = nodes.NODE_CLASS_MAPPINGS['VAEEncode']()
                    refined_latent = encode_node.encode(vae, clamped_final_image)[0]
                
                if not isinstance(refined_latent, dict): refined_latent = {"samples": refined_latent}
            
            if refined_latent is None:
                h_final, w_final = final_image.shape[1], final_image.shape[2]
                refined_latent = {"samples": torch.zeros((1, 4, h_final // 8, w_final // 8), dtype=torch.float32, device=device)}

            return (refined_latent, final_image.cpu())

        except model_management.InterruptProcessingException:
            print("Refiner cancelled by user")
            self.cached_input_tensor = None
            raise
        except Exception as e:
            import traceback
            print(f"FATAL Error in Refiner: {e}")
            traceback.print_exc()
            dummy_latent = {"samples": torch.zeros((1, 4, 64, 64))}
            dummy_image = torch.zeros((1, 64, 64, 3))
            if is_latent_input and latent is not None: dummy_latent = latent
            if is_image_input and image is not None: dummy_image = image
            return (dummy_latent, dummy_image)
    def apply_auto_tone_correction_gpu(self, image_tensor):
        try:
            check_for_interruption()
            
            prepared_image = self.prepare_source_dynamics(image_tensor)
            final_image = self.finalize_tone_correction(prepared_image)
            
            return final_image

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            import traceback
            print(f"FATAL Error in auto tone correction: {e}")
            traceback.print_exc()
            return image_tensor
    def prepare_source_dynamics(self, image_tensor):
        try:
            check_for_interruption()
            original_device = image_tensor.device

            recovered_tensor = self._recover_channel_clipping_gpu(image_tensor)
            
            image_np = recovered_tensor.cpu().numpy().astype(np.float64)
            prepared_images = []

            for i in range(image_np.shape[0]):
                img = image_np[i]
                
                lab_img = rgb2lab(img)
                l_channel = lab_img[:, :, 0]
                l_channel_normalized = l_channel / 100.0
                
                shadow_clip_point = np.percentile(l_channel_normalized, 1)
                highlight_clip_point = np.percentile(l_channel_normalized, 99)
                
                if shadow_clip_point < 0.02 or highlight_clip_point > 0.98:
                    prepared_l_normalized = self._prevent_clipping_preserve_detail(
                        l_channel_normalized, shadow_clip_point, highlight_clip_point
                    )
                else:
                    prepared_l_normalized = l_channel_normalized
                
                prepared_l = prepared_l_normalized * 100.0
                prepared_lab = lab_img.copy()
                prepared_lab[:, :, 0] = prepared_l
                
                prepared_rgb = lab2rgb(prepared_lab)
                prepared_images.append(prepared_rgb)

            final_np = np.stack(prepared_images).astype(np.float32)
            final_tensor = torch.from_numpy(final_np).to(original_device)
            
            return torch.clamp(final_tensor, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            import traceback
            print(f"FATAL Error during source dynamics preparation: {e}")
            traceback.print_exc()
            return image_tensor    
    def finalize_tone_correction(self, image_tensor):
        if not SKIMAGE_AVAILABLE:
            print("-------------------------------------------------------------------------------------------------")
            print("WARNING: scikit-image library not found.")
            print("Please install it by running: pip install scikit-image")
            print("Skipping Final Tone Correction.")
            print("-------------------------------------------------------------------------------------------------")
            return image_tensor

        try:
            check_for_interruption()
            print("Finalize tone correction is running")
            original_device = image_tensor.device
            print(f"DEBUG: Input to finalize_tone_correction - min: {image_tensor.min():.3f}, max: {image_tensor.max():.3f}")
        
            image_np = image_tensor.cpu().numpy().astype(np.float64)
            corrected_images = []

            for i in range(image_np.shape[0]):
                img = image_np[i]
                
                lab_img = rgb2lab(img)
                l_channel = lab_img[:, :, 0]
                a_channel = lab_img[:, :, 1]
                b_channel = lab_img[:, :, 2]
                l_channel_normalized = l_channel / 100.0
                
                shadow_clip_point = np.percentile(l_channel_normalized, 1)
                highlight_clip_point = np.percentile(l_channel_normalized, 99)
                print(f"DEBUG: Running FULL tone correction (shadow: {shadow_clip_point:.3f}, highlight: {highlight_clip_point:.3f})")

                corrected_l_normalized = self._prevent_clipping_preserve_detail(
                    l_channel_normalized, shadow_clip_point, highlight_clip_point
                )

                corrected_l_normalized = self._apply_smooth_post_processing(corrected_l_normalized)
                safe_shadow_level = 0.02

                corrected_l_normalized = self._intelligently_deepen_blacks(
                    corrected_l_normalized, a_channel, b_channel, safe_shadow_level
                )

                print(f"DEBUG: After full correction - min: {np.min(corrected_l_normalized):.3f}, max: {np.max(corrected_l_normalized):.3f}")
                
                corrected_l = corrected_l_normalized * 100.0
                corrected_lab = lab_img.copy()
                corrected_lab[:, :, 0] = corrected_l
                
                corrected_rgb = lab2rgb(corrected_lab)
                corrected_images.append(corrected_rgb)

            final_np = np.stack(corrected_images).astype(np.float32)
            final_tensor = torch.from_numpy(final_np).to(original_device)
            
            return torch.clamp(final_tensor, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            import traceback
            print(f"FATAL Error during final tone correction: {e}")
            traceback.print_exc()
            return image_tensor
    def apply_clipping_protection(self, image_bchw):
        try:
            check_for_interruption()
            
            protected_image = torch.clamp(image_bchw, 0.02, 0.98)
            
            image_lab = kornia.color.rgb_to_lab(protected_image)
            l_channel = image_lab[:, 0:1] / 100.0
            
            highlight_mask = (l_channel > 0.95).float()
            shadow_mask = (l_channel < 0.05).float()
            
            if torch.sum(highlight_mask) > 0:
                l_channel = torch.where(highlight_mask > 0, torch.clamp(l_channel, 0, 0.95), l_channel)
            
            if torch.sum(shadow_mask) > 0:
                l_channel = torch.where(shadow_mask > 0, torch.clamp(l_channel, 0.05, 1), l_channel)
            
            corrected_lab = torch.cat([l_channel * 100.0, image_lab[:, 1:3]], dim=1)
            corrected_rgb = kornia.color.lab_to_rgb(corrected_lab)
            
            return torch.clamp(corrected_rgb, 0.0, 1.0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Warning: Clipping protection failed: {e}")
            return torch.clamp(image_bchw, 0.0, 1.0)
    def _prevent_clipping_preserve_detail(self, luminance, shadow_clip, highlight_clip):
        """
        Prevents clipping by lifting truly crushed blacks AND compressing truly blown-out highlights.
        --- DEFINITIVE VERSION ---
        This version applies the same precise logic to both shadows and highlights, ensuring that
        only pixels on the verge of clipping (near 0.0 or 1.0) are adjusted, while leaving
        all other tones untouched.
        """
        result = luminance.copy()
        
        # --- Define Symmetrical Levels ---
        
        # Shadow Levels
        safe_shadow_level = 0.02      # The target we lift crushed blacks TO.
        shadow_clipping_threshold = 0.005 # The strict mask to identify WHAT IS a crushed black.

        # Highlight Levels
        safe_highlight_level = 0.93     # The target we compress blown-out whites TO.
        highlight_clipping_threshold = 0.995 # The strict mask to identify WHAT IS a blown-out white.

        # --- Corrected Shadow Lifting Logic ---
        # Entry Condition: Is the image dark enough to potentially have crushed blacks?
        if shadow_clip < safe_shadow_level:
            # Selection Mask: Is a pixel *actually* crushed below the strict threshold?
            pixels_to_lift_mask = result < shadow_clipping_threshold
            
            if np.any(pixels_to_lift_mask):
                original_values = result[pixels_to_lift_mask]
                lift_needed = safe_shadow_level - original_values
                blend_weight = (1.0 - (original_values / safe_shadow_level)) ** 1.5
                result[pixels_to_lift_mask] = original_values + (lift_needed * blend_weight)

        # --- Symmetrical Highlight Compression Logic ---
        # Entry Condition: Is the image bright enough to potentially have blown-out highlights?
        if highlight_clip > safe_highlight_level:
            # Selection Mask: Is a pixel *actually* blown-out above the strict threshold?
            pixels_to_compress_mask = result > highlight_clipping_threshold

            if np.any(pixels_to_compress_mask):
                original_values = result[pixels_to_compress_mask]

                # Calculate how much we need to reduce the highlights to bring them to the safe level.
                compression_needed = original_values - safe_highlight_level

                # Create a blend weight. The effect is strongest for pure white (1.0) and fades to nothing
                # as the pixel value approaches the clipping threshold.
                # This prevents a hard edge at the boundary of the mask.
                blend_weight = ((original_values - highlight_clipping_threshold) / (1.0 - highlight_clipping_threshold)) ** 1.5
                
                # Apply the targeted compression.
                result[pixels_to_compress_mask] = original_values - (compression_needed * blend_weight)
        
        return np.clip(result, 0, 1)
    def _recover_channel_clipping_gpu(self, image_tensor_bhwc):
        """
        Recovers detail from pre-clipped R, G, or B channels in the source image.
        This runs FIRST, before any other tonal adjustments.
        """
        try:
            # --- Setup ---
            original_device = image_tensor_bhwc.device
            image_bchw = image_tensor_bhwc.permute(0, 3, 1, 2)
            
            highlight_clipping_threshold = 0.995 # Identifies what IS clipped.
            safe_highlight_level = 0.98         # Defines the target to recover TO.
            
            # --- Identify Clipped Pixels ---
            # Create a mask for any pixel where any channel is blown out.
            clipping_mask = torch.any(image_bchw > highlight_clipping_threshold, dim=1, keepdim=True)

            if not torch.any(clipping_mask):
                return image_tensor_bhwc # Return original if no clipping is found.

            # --- Recover Detail ---
            # Convert the entire image to HSV to manipulate brightness (Value) without affecting color (Hue).
            hsv_image = kornia.color.rgb_to_hsv(image_bchw)
            h, s, v = hsv_image[:, 0:1], hsv_image[:, 1:2], hsv_image[:, 2:3]
            
            # Isolate the HSV components for only the pixels we need to fix.
            clipped_h = h[clipping_mask.expand_as(h)].view(1, 1, -1, 1)
            clipped_s = s[clipping_mask.expand_as(s)].view(1, 1, -1, 1)
            clipped_v = v[clipping_mask.expand_as(v)].view(1, 1, -1, 1)

            # --- Binary Search for the correct brightness ---
            # We search for the highest possible 'Value' that keeps all RGB channels safely below our target.
            low = torch.zeros_like(clipped_v) # The darkest possible value is 0.
            high = clipped_v                  # The brightest possible is its current value.

            for _ in range(8): # 8 iterations is enough for 8-bit precision.
                mid_v = (low + high) / 2.0
                # Create a test image using the mid-point Value
                test_hsv = torch.cat([clipped_h, clipped_s, mid_v], dim=1)
                test_rgb = kornia.color.hsv_to_rgb(test_hsv)
                
                # Check if this test Value results in an RGB color that is safely in gamut.
                is_safe = torch.all(test_rgb <= safe_highlight_level, dim=1, keepdim=True)
                
                # If it's safe, we can afford to try a brighter Value.
                low = torch.where(is_safe, mid_v, low)
                # If it's not safe, the Value is still too high.
                high = torch.where(is_safe, high, mid_v)

            # 'low' now holds the optimal recovered brightness. Update the main image's Value channel.
            v.masked_scatter_(clipping_mask.expand_as(v), low.flatten())
            
            # Convert the corrected HSV image back to RGB.
            recovered_bchw = kornia.color.hsv_to_rgb(torch.cat([h, s, v], dim=1))
            
            # Return in the original BHWC format.
            return recovered_bchw.permute(0, 2, 3, 1)

        except Exception as e:
            print(f"FATAL Error during channel highlight recovery: {e}")
            return torch.clamp(image_tensor_bhwc, 0.0, 1.0)
    def _intelligently_deepen_blacks(self, luminance, lab_a, lab_b, safe_shadow_level):
        """
        Deepens blacks using a smoothed, continuous, and dynamically modulated strength map.
        --- DEFINITIVE VERSION ---
        This version fixes harsh masking artifacts by replacing the binary mask with a
        smooth, feathered weight map and by smoothing the modulation maps themselves.
        """
        result = luminance.copy()
        
        analysis = self._analyze_black_deepening_requirements(luminance, safe_shadow_level)
        
        safe_black_start = safe_shadow_level
        safe_black_end = np.clip(0.35 + analysis['black_end_adjustment'], 0.25, 0.45)
        
        # --- LOGGING (Unchanged) ---
        print("=" * 60)
        print("BLACK DEEPENING ANALYSIS (DEFINITIVE - Smoothed Masks)")
        print("=" * 60)
        # ... (rest of the initial logging)
        
        # --- THE FIX 1: Create a Smooth, Continuous Weight Mask ---
        # Instead of a hard binary mask, we create a feathered "bell curve" mask.
        center_of_range = (safe_black_start + safe_black_end) / 2.0
        width_of_range = (safe_black_end - safe_black_start)
        # Calculate how far each pixel is from the center of our target range.
        distance_from_center = np.abs(luminance - center_of_range)
        # Create a smooth falloff. A pixel at the center has a weight of 1.0.
        # The weight drops to 0.0 as it approaches the start or end of the range.
        range_weight_mask = np.clip(1.0 - (distance_from_center / (width_of_range / 2.0)), 0.0, 1.0)
        # Apply a power to the curve to control the feathering.
        range_weight_mask = np.power(range_weight_mask, 0.8)

        if not np.any(range_weight_mask > 0):
             return result # Exit if no pixels are meaningfully in range

        # --- THE FIX 2: Smooth the Modulation Maps ---
        if SCIPY_AVAILABLE:
            # Calculate the raw modulation maps as before
            blurred_luminance = ndimage.gaussian_filter(luminance, sigma=1.5)
            detail_map = np.abs(luminance - blurred_luminance)
            flatness_modulation_raw = 1.6 - np.tanh(detail_map * 20.0) * 1.1
            # NOW, SMOOTH THE MASK ITSELF to eliminate pixel-level noise and create a regional effect.
            flatness_modulation = ndimage.gaussian_filter(flatness_modulation_raw, sigma=2.0)
        else:
            flatness_modulation = np.ones_like(result)

        saturation = np.sqrt(lab_a**2 + lab_b**2)
        color_protection_raw = 1.0 - np.tanh(saturation / 40.0) * 0.6
        # SMOOTH THE COLOR MASK as well for consistency.
        color_protection = ndimage.gaussian_filter(color_protection_raw, sigma=2.0)
        
        # --- Combine and Apply ---
        base_alpha = analysis['max_deepening_strength'] * analysis['deepening_strength_multiplier']
        
        # The final strength is a combination of all our smooth masks.
        final_alpha_map = base_alpha * range_weight_mask * flatness_modulation * color_protection
        final_alpha_map = final_alpha_map * 1.5  # Boost overall strength
        final_alpha_map = np.clip(final_alpha_map, 0.0, 0.95)

        # Calculate the ideal 'deepened' look for the entire image
        range_compression_factor = np.clip(1.0 - (analysis['deepening_strength_multiplier'] - 1.0) * 0.25, 0.75, 1.0)
        black_range = safe_black_end - safe_black_start
        compressed_black_range = black_range * range_compression_factor
        normalized_position = np.clip((luminance - safe_black_start) / black_range, 0.0, 1.0)
        curved_position = np.power(normalized_position, analysis['deepening_curve_power'])
        target_deepened_values = safe_black_start + (curved_position * compressed_black_range)
        
        # Linearly interpolate between the original and the ideal look using our final smooth map.
        # This is a per-pixel, feathered blend that avoids all hard edges.
        final_result = (luminance * (1 - final_alpha_map)) + (target_deepened_values * final_alpha_map)

        print(f"DEBUG: Black deepening effect:")
        print(f"  Original range: {np.min(luminance):.3f} - {np.max(luminance):.3f}")
        print(f"  Target range: {np.min(target_deepened_values):.3f} - {np.max(target_deepened_values):.3f}")
        print(f"  Final range: {np.min(final_result):.3f} - {np.max(final_result):.3f}")
        print(f"  Alpha map range: {np.min(final_alpha_map):.3f} - {np.max(final_alpha_map):.3f}")
        print(f"  Pixels getting >50% effect: {np.mean(final_alpha_map > 0.5):.3f}")
        print(f"\nDeepening Statistics:")
        print(f"  Base Alpha: {base_alpha:.3f}")
        print(f"  Final Blend Alpha - Min: {np.min(final_alpha_map):.3f}, Avg: {np.mean(final_alpha_map):.3f}, Max: {np.max(final_alpha_map):.3f}")

        return final_result
    def _analyze_black_deepening_requirements(self, luminance, safe_shadow_level):
        """
        Analyzes the image to determine optimal black deepening parameters.
        --- REVISED v2 ---
        This version adjusts the weighting to be less conservative on high-contrast images
        that still have "muddy" shadow regions, allowing for a stronger base effect.
        """
        try:
            lum_np = luminance.cpu().numpy() if hasattr(luminance, 'cpu') else luminance
            
            # --- Core Image Metrics ---
            lum_std = np.std(lum_np)
            lum_mean = np.mean(lum_np)
            contrast_ratio = lum_std / (lum_mean + 1e-8)
            
            true_blacks = np.mean(lum_np < 0.05)
            dark_shadows = np.mean(lum_np < 0.15)
            medium_shadows = np.mean((lum_np >= 0.15) & (lum_np < 0.3))
            
            midtone_concentration = np.mean((lum_np >= 0.3) & (lum_np <= 0.7))
            
            # --- DYNAMIC FACTORS ---
            
            # Contrast Factor: Penalizes high contrast, but less aggressively than before.
            contrast_sigmoid = 1.0 / (1.0 + np.exp((contrast_ratio - 0.4) * 10.0))
            contrast_factor = contrast_sigmoid * 0.8 + 0.2

            # Black Deficit Factor: Stronger signal when blacks are needed.
            optimal_black_percentage = 0.075
            distance_from_optimal = np.abs(true_blacks - optimal_black_percentage)
            black_deficit_factor = np.exp(-np.power(distance_from_optimal / 0.1, 2))

            midtone_optimal = 0.45
            midtone_distance = np.abs(midtone_concentration - midtone_optimal)
            midtone_factor = np.exp(-np.power(midtone_distance / 0.25, 2))
            
            # THE FIX: Adjust the final weighting. 'black_deficit' now has the strongest voice,
            # and 'contrast' has less power to veto the operation.
            combined_factor = np.clip((
                contrast_factor * 0.20 +
                black_deficit_factor * 0.50 +
                midtone_factor * 0.30
            ), 0.0, 1.0)
            
            # --- Final Parameter Calculation ---
            strength_multiplier = 0.8 + (combined_factor * 1.2)
            
            contrast_restriction = -0.08 * np.power(np.clip((contrast_ratio - 0.45) / 0.5, 0.0, 1.0), 1.5)
            midtone_extension = 0.1 * np.power(np.clip((midtone_concentration - 0.4) / 0.3, 0.0, 1.0), 0.8)
            black_end_adjustment = contrast_restriction + midtone_extension
            black_end_adjustment = np.clip(black_end_adjustment, -0.1, 0.12)
            
            shadow_normalized = np.clip(dark_shadows, 0.05, 0.45)
            curve_power = 1.1 + 0.7 * np.power((0.45 - shadow_normalized) / 0.4, 0.8)
            
            max_deepening_strength = 0.20 + (combined_factor * 0.50)
            max_deepening_strength = np.clip(max_deepening_strength, 0.15, 0.7)
            print(f"DEBUG: Black Analysis - contrast_ratio: {float(contrast_ratio):.3f}")
            print(f"DEBUG: Black Analysis - true_blacks: {float(true_blacks):.3f} (want ~0.075)")
            print(f"DEBUG: Black Analysis - combined_factor: {float(combined_factor):.3f}")
            print(f"DEBUG: Black Analysis - strength_multiplier: {float(strength_multiplier):.3f}")
            print(f"DEBUG: Black Analysis - max_deepening_strength: {float(max_deepening_strength):.3f}")
            return {
                'contrast_ratio': float(contrast_ratio),
                'true_blacks_percentage': float(true_blacks),
                'dark_shadows_percentage': float(dark_shadows),
                'medium_shadows_percentage': float(medium_shadows),
                'midtone_concentration': float(midtone_concentration),
                'deepening_strength_multiplier': float(strength_multiplier),
                'black_end_adjustment': float(black_end_adjustment),
                'deepening_curve_power': float(curve_power),
                'max_deepening_strength': float(max_deepening_strength),
                'combined_factor': float(combined_factor),
                'contrast_factor': float(contrast_factor),
                'black_deficit_factor': float(black_deficit_factor),
                'midtone_factor': float(midtone_factor)
            }
            
        except Exception as e:
            print(f"Error in black deepening analysis: {e}")
            # Return safe defaults
            return {'deepening_strength_multiplier': 1.0, 'black_end_adjustment': 0.0, 'deepening_curve_power': 1.2, 'max_deepening_strength': 0.4}
    def _preserve_black_detail(self, luminance, safe_shadow_level, safe_black_end):

        
        if SCIPY_AVAILABLE:
            
            black_region_mask = (luminance >= safe_shadow_level) & (luminance <= safe_black_end)
            
            if np.any(black_region_mask):
                sigma = 1.0
                blurred = ndimage.gaussian_filter(luminance, sigma=sigma)
                detail_enhancement = (luminance - blurred) * 0.3
                
                result = luminance.copy()
                result[black_region_mask] += detail_enhancement[black_region_mask]
                
                return np.clip(result, safe_shadow_level, 1.0)
            
        else:
            return self._simple_black_detail_preserve(luminance, safe_shadow_level, safe_black_end)
        
        return luminance

    def _simple_black_detail_preserve(self, luminance, safe_shadow_level, safe_black_end):
    
        
        result = luminance.copy()
        black_region_mask = (result >= safe_shadow_level) & (result <= safe_black_end)
        
        if np.any(black_region_mask):
            padded = np.pad(result, 2, mode='reflect')
            
            for y in range(2, padded.shape[0] - 2):
                for x in range(2, padded.shape[1] - 2):
                    orig_y, orig_x = y - 2, x - 2
                    if orig_y < result.shape[0] and orig_x < result.shape[1] and black_region_mask[orig_y, orig_x]:
                        local_avg = np.mean(padded[y-1:y+2, x-1:x+2])
                        detail_diff = result[orig_y, orig_x] - local_avg
                        result[orig_y, orig_x] += detail_diff * 0.1
        
        return np.clip(result, safe_shadow_level, 1.0)
    def _apply_smooth_post_processing(self, luminance):

        
        if SCIPY_AVAILABLE:
            min_dim = min(luminance.shape[0], luminance.shape[1])
            
            if min_dim > 1024:
                sigma = 1.0
            elif min_dim > 512:
                sigma = 0.7
            else:
                sigma = 0.4
                
            smoothed = ndimage.gaussian_filter(luminance, sigma=sigma, mode='reflect')
            
            blend_factor = 0.15
            result = luminance * (1 - blend_factor) + smoothed * blend_factor
            
            return result
        else:
            return self._simple_neighbor_smooth(luminance)

    def _simple_neighbor_smooth(self, luminance):
   
        
        padded = np.pad(luminance, 1, mode='reflect')
        
        result = luminance.copy()
        
        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if y-1 < result.shape[0] and x-1 < result.shape[1]:
                    center = padded[y, x] * 0.6
                    neighbors = (padded[y-1:y+2, x-1:x+2].sum() - padded[y, x]) * 0.4 / 8
                    result[y-1, x-1] = center + neighbors
        
        return result
   
   
    def _tint_in_lab(self, image_bchw, target_color_rgb, mask):

        try:
            if not isinstance(target_color_rgb, torch.Tensor):
                target_color_rgb = torch.tensor(target_color_rgb, device=image_bchw.device, dtype=image_bchw.dtype)
            
            source_lab = kornia.color.rgb_to_lab(image_bchw)
            target_color_image = target_color_rgb.view(1, 3, 1, 1).expand_as(image_bchw)
            target_lab = kornia.color.rgb_to_lab(target_color_image)
            
            target_a = target_lab[:, 1:2]
            target_b = target_lab[:, 2:3]

            blended_a = torch.lerp(source_lab[:, 1:2], target_a, mask)
            blended_b = torch.lerp(source_lab[:, 2:3], target_b, mask)
            
            final_lab = torch.cat([source_lab[:, 0:1], blended_a, blended_b], dim=1)
            
            return kornia.color.lab_to_rgb(final_lab)
            
        except Exception as e:
            print(f"Error in _tint_in_lab: {e}")
            return image_bchw
   
    def apply_mood_and_lighting_transfer(self, image_bchw, subject_mask, depth_map, preset_data, strength):
  
        try:
            check_for_interruption()
            if not preset_data or strength == 0.0:
                return image_bchw
            
            for name, data in mood_presets.PRESETS.items():
                if data == preset_data:
                    break
            

            h, w = image_bchw.shape[-2:]
            device = image_bchw.device
            scale_factor = min(h, w) / 1024.0
            
            full_effect_image = self._apply_full_mood_effects(image_bchw, preset_data, scale_factor, depth_map)
            
            foreground_50_image = self._apply_reduced_mood_effects(image_bchw, preset_data, scale_factor, 0.4)
            
            mask_system = self._create_simple_mask_system(subject_mask, scale_factor, device, depth_map)
            
            final_result = self._merge_with_inward_feather(
                full_effect_image, foreground_50_image, mask_system
            )
            
            final_image = torch.lerp(image_bchw, final_result, strength)
            print(f"DEBUG: Mood preset output BEFORE clipping protection - min: {final_image.min():.3f}, max: {final_image.max():.3f}")
            final_image = self.apply_clipping_protection(final_image)

            return final_image

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: Error applying mood: {e}")
            return image_bchw
    def _extract_colors_with_kmeans(self, image_bchw, mask, n_clusters, device):

        try:

            mask_bool = mask.squeeze(0).squeeze(0) > 0.5

            if mask_bool.sum() == 0:
                return []

            pixels = image_bchw.squeeze(0).permute(1, 2, 0)[mask_bool]

            pixels = pixels.to(torch.float32)

            
            if not SKLEARN_AVAILABLE:
                print("-------------------------------------------------------------------------------------------------")
                print("WARNING: scikit-learn library not found.")
                print("Please install it running requirements.txt")
                print("-------------------------------------------------------------------------------------------------")
                return []
            
            actual_clusters = min(n_clusters, pixels.shape[0])
            if actual_clusters == 0:
                return []

            kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init='auto').fit(pixels.cpu().numpy())
            
            colors_rgb = kmeans.cluster_centers_.tolist()

            return colors_rgb

        except Exception as e:
            print(f"Warning: K-Means color extraction failed: {e}")
            return []
    
    def _apply_full_mood_effects(self, image_bchw, preset_data, scale_factor, depth_map):
   
        try:
            check_for_interruption()
            
            h, w = image_bchw.shape[-2:]
            device = image_bchw.device
            
            background_mask = torch.ones((1, 1, h, w), device=device)
            
            if depth_map is not None:
                if depth_map.shape[-2:] != (h, w):
                    depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)
                
                far_regions = (1.0 - depth_map).pow(1.2)
                background_mask = background_mask * (0.7 + 0.3 * far_regions)
            
            subject_mask_for_analysis = torch.ones((1, 1, h, w), device=device)
            image_analysis = self._analyze_image_characteristics(image_bchw, subject_mask_for_analysis, scale_factor)
            
            mask_system = {'background_mask': background_mask}
            
            result = image_bchw.clone()
            
            color_palette = preset_data.get("color_palette", {})
            if color_palette:
                result = self._apply_preset_color_grading(result, color_palette, image_analysis, mask_system)
            
            temperature_shift = preset_data.get("temperature_shift", "none")
            if temperature_shift != "none":
                result = self._apply_preset_temperature(result, temperature_shift, image_analysis, mask_system)
            
            brightness_shift = preset_data.get("brightness_shift", "enhance_existing")
            if brightness_shift != "enhance_existing":
                result = self._apply_preset_brightness(result, brightness_shift, image_analysis, mask_system)
            
            atmospheric_effects = preset_data.get("atmospheric_effects", {})
            if atmospheric_effects:
                result = self._apply_preset_atmospheric_effects(result, atmospheric_effects, image_analysis, mask_system, scale_factor)
            
            result = self.apply_clipping_protection(result)
            return result
            
        except Exception as e:
            print(f"Error in full mood effects: {e}")
            return image_bchw

    def _apply_reduced_mood_effects(self, image_bchw, preset_data, scale_factor, reduction_factor):
      
        try:
            check_for_interruption()
            
            h, w = image_bchw.shape[-2:]
            device = image_bchw.device
            full_mask = torch.ones((1, 1, h, w), device=device)
            
            image_analysis = self._analyze_image_characteristics(image_bchw, full_mask, scale_factor)
            mask_system = {'background_mask': full_mask}
            
            result = image_bchw.clone()
            
            color_palette = preset_data.get("color_palette", {})
            if color_palette:
                result = self._apply_preset_color_grading(result, color_palette, image_analysis, mask_system)
            
            temperature_shift = preset_data.get("temperature_shift", "none")
            if temperature_shift != "none":
                result = self._apply_preset_temperature(result, temperature_shift, image_analysis, mask_system)
            
            brightness_shift = preset_data.get("brightness_shift", "enhance_existing")
            if brightness_shift != "enhance_existing":
                result = self._apply_preset_brightness(result, brightness_shift, image_analysis, mask_system)
            
            
            final_result = torch.lerp(image_bchw, result, reduction_factor)
            
            final_result = self.apply_clipping_protection(final_result)
            return final_result
            
        except Exception as e:
            print(f"Error in reduced mood effects: {e}")
            return image_bchw    
    def _create_simple_mask_system(self, subject_mask, scale_factor, device, depth_map):
      
        try:
            check_for_interruption()
            
            h, w = subject_mask.shape[-2:]
            
            if subject_mask.device != device:
                subject_mask = subject_mask.to(device)
            
            original_smooth_mask = torch.clamp(subject_mask, 0.0, 1.0)
            
            core_mask = torch.pow(original_smooth_mask, 1.5)
            
            soft_mask = original_smooth_mask
            
            if scale_factor > 1.0:
                wide_blur = max(8.0 * scale_factor, 4.0)
                wide_kernel = int(wide_blur * 2) | 1
                wide_mask = kornia.filters.gaussian_blur2d(
                    original_smooth_mask, (wide_kernel, wide_kernel), (wide_blur, wide_blur)
                )
            else:
                wide_mask = original_smooth_mask
            
            background_mask = torch.clamp(1.0 - wide_mask, 0.0, 1.0)
            
            if depth_map is not None:
                if depth_map.device != device:
                    depth_map = depth_map.to(device)
                if depth_map.shape[-2:] != (h, w):
                    depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)
                
                depth_map = torch.clamp(depth_map, 0.0, 1.0)
                
                near_regions = depth_map.pow(1.8)
                far_regions = (1.0 - depth_map).pow(1.2)
                
                depth_enhanced_core = core_mask * (0.9 + 0.1 * near_regions)
                depth_enhanced_soft = soft_mask * (0.95 + 0.05 * near_regions)
                
            else:
                depth_enhanced_core = core_mask
                depth_enhanced_soft = soft_mask
                near_regions = torch.ones_like(core_mask)
                far_regions = torch.zeros_like(core_mask)
            
            return {
                'core_mask': torch.clamp(depth_enhanced_core, 0.0, 1.0),
                'soft_mask': torch.clamp(depth_enhanced_soft, 0.0, 1.0),
                'background_mask': background_mask,
                'near_regions': near_regions,
                'far_regions': far_regions,
                'wide_mask': wide_mask,
                'original_mask': original_smooth_mask
            }
            
        except Exception as e:
            print(f"Error creating mask system: {e}")
            safe_mask = torch.clamp(subject_mask, 0.0, 1.0)
            return {
                'core_mask': torch.pow(safe_mask, 1.5), 'soft_mask': safe_mask,
                'background_mask': torch.clamp(1.0 - safe_mask, 0.0, 1.0), 
                'wide_mask': safe_mask, 'original_mask': safe_mask,
                'near_regions': torch.ones_like(safe_mask),
                'far_regions': torch.zeros_like(safe_mask)
            }

    def _merge_with_inward_feather(self, full_effect_image, foreground_50_image, mask_system):
     
        try:
            check_for_interruption()
            
            soft_mask = mask_system['soft_mask']
            
            result = full_effect_image.clone()
            
            result = torch.lerp(result, foreground_50_image, soft_mask)
            
            
            edge_threshold = 0.9
            contaminated_edges = torch.clamp((edge_threshold - soft_mask) / edge_threshold, 0.0, 1.0)
            contaminated_edges = contaminated_edges * soft_mask
            
            correction_strength = 0.6
            
            result = torch.lerp(result, full_effect_image, contaminated_edges * correction_strength)
            
            return torch.clamp(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error replacing edge artifacts: {e}")
            return foreground_50_image
    
    def _analyze_image_characteristics(self, image_bchw, subject_mask, scale_factor, generated_mood_type=None):
      
        try:
            check_for_interruption()
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            l_channel = image_lab[:, 0:1] / 100.0
            a_channel = image_lab[:, 1:2]
            b_channel = image_lab[:, 2:3]
            
            overall_brightness = torch.mean(l_channel)
            overall_contrast = torch.std(l_channel)
            color_temp = torch.mean(b_channel) / 100.0
            
            blur_radius = max(8.0 * scale_factor, 4.0)
            kernel_size = int(blur_radius * 2) | 1
            blurred_l = kornia.filters.gaussian_blur2d(
                l_channel, (kernel_size, kernel_size), (blur_radius, blur_radius)
            )
            
            highlight_threshold = torch.quantile(blurred_l.flatten(), 0.85)
            highlight_mask = torch.clamp((blurred_l - highlight_threshold) / (1.0 - highlight_threshold), 0, 1)

            highlight_intensity = torch.mean(blurred_l[highlight_mask > 0.1]) if torch.sum(highlight_mask) > 0.1 else 0.5
            
            shadow_threshold = torch.quantile(blurred_l.flatten(), 0.25)
            shadow_mask = (blurred_l < shadow_threshold).float()
            shadow_depth = torch.mean(blurred_l[shadow_mask > 0.5]) if torch.sum(shadow_mask) > 0 else 0.3
            
            background_mask = 1.0 - subject_mask
            if torch.sum(background_mask) > 0:
                bg_brightness = torch.mean(l_channel[background_mask > 0.5])
            else:
                bg_brightness = overall_brightness
            
            if torch.sum(subject_mask) > 0:
                fg_brightness = torch.mean(l_channel[subject_mask > 0.5])
                fg_contrast = torch.std(l_channel[subject_mask > 0.5])
            else:
                fg_brightness = overall_brightness
                fg_contrast = overall_contrast

            analysis_result = {
                'overall_brightness': overall_brightness.item(),
                'overall_contrast': overall_contrast.item(),
                'color_temperature': color_temp.item(),
                'highlight_intensity': highlight_intensity.item() if isinstance(highlight_intensity, torch.Tensor) else highlight_intensity,
                'shadow_depth': shadow_depth.item() if isinstance(shadow_depth, torch.Tensor) else shadow_depth,
                'bg_brightness': bg_brightness.item(),
                'fg_brightness': fg_brightness.item(),
                'fg_contrast': fg_contrast.item(),
                'highlight_mask': highlight_mask,
                'shadow_mask': shadow_mask,
                'is_bright_image': overall_brightness > 0.6,
                'is_dark_image': overall_brightness < 0.3,
                'is_high_contrast': overall_contrast > 0.25,
                'is_warm': color_temp > 0.1,
                'is_cool': color_temp < -0.1,
                'scale_factor': scale_factor
            }
            if generated_mood_type:
                analysis_result['mood_type'] = generated_mood_type
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in image analysis: {e}")
            return self._get_default_analysis(scale_factor)

    def _get_default_analysis(self, scale_factor):
        
        device = torch.device('cpu')
        return {
            'overall_brightness': 0.5, 'overall_contrast': 0.2, 'color_temperature': 0.0,
            'highlight_intensity': 0.7, 'shadow_depth': 0.3, 'bg_brightness': 0.5,
            'bg_highlight_ratio': 0.1, 'fg_brightness': 0.5, 'fg_contrast': 0.2,
            'highlight_mask': torch.zeros((1, 1, 64, 64), device=device),
            'shadow_mask': torch.zeros((1, 1, 64, 64), device=device),
            'is_bright_image': False, 'is_dark_image': False, 'is_high_contrast': False,
            'is_warm': False, 'is_cool': False, 'scale_factor': scale_factor
        }
    
    
    def _apply_preset_brightness(self, image_bchw, brightness_shift, image_analysis, mask_system):
     
        try:
            check_for_interruption()
            background_mask = mask_system['background_mask']
            
            adjustment = 0.0
            if isinstance(brightness_shift, str):
                shift_mapping = {"slightly_moodier": -8.0, "much_darker": -18.0, "moodier": -12.0, "darker_but_vibrant": -10.0}
                adjustment = shift_mapping.get(brightness_shift, 0.0)
            elif isinstance(brightness_shift, (int, float)):
                adjustment = brightness_shift

            if adjustment == 0.0:
                return image_bchw
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            l_channel = image_lab[:, 0:1]
            l_normalized = l_channel / 100.0
            protection_curve = 4.0 * l_normalized * (1.0 - l_normalized)
            l_adjusted = l_channel + (adjustment * background_mask * protection_curve)
            adjusted_lab = torch.cat([l_adjusted, image_lab[:, 1:3]], dim=1)
            adjusted_rgb = kornia.color.lab_to_rgb(adjusted_lab)
            return torch.clamp(adjusted_rgb, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in preset brightness: {e}")
            return image_bchw

    def _apply_preset_atmospheric_effects(self, image_bchw, atmospheric_effects, image_analysis, mask_system, scale_factor):
     
        try:
            check_for_interruption()
            
            effect_type = atmospheric_effects.get("type", "clean_natural")
            glow_colors = atmospheric_effects.get("glow_colors", [])
            complex_glows = atmospheric_effects.get("complex_glows", False)
            
            intensity_multiplier = atmospheric_effects.get("intensity_multiplier", 1.0)
            saturation_boost = atmospheric_effects.get("saturation_boost", 1.0)
            coverage_multiplier = atmospheric_effects.get("coverage_multiplier", 1.0)
            
            if not glow_colors:
                return image_bchw
            
            result = image_bchw.clone()
            device = image_bchw.device
            h, w = image_bchw.shape[-2:]
            background_mask = mask_system['background_mask']
            
            highlight_mask = image_analysis['highlight_mask'].to(device)
            bg_highlights = highlight_mask * background_mask
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            l_channel = image_lab[:, 0:1]
            l_normalized = l_channel / 100.0
            
            deep_shadow_threshold = 0.15
            
            deep_shadow_protection_map = torch.clamp(1.0 - (l_normalized / deep_shadow_threshold), 0.0, 1.0)

            final_glow_application_mask = background_mask * (1.0 - deep_shadow_protection_map)
            
            
            if effect_type == "clean_natural":
                for i, glow_color in enumerate(glow_colors):
                    result = self._apply_natural_glow_from_preset(
                        result, glow_color, bg_highlights, final_glow_application_mask, scale_factor, i
                    )
                    
            elif effect_type == "warm_cinematic":
                for i, glow_color in enumerate(glow_colors):
                    result = self._apply_cinematic_glow_from_preset(
                        result, glow_color, bg_highlights, final_glow_application_mask, scale_factor, i
                    )
                    
            elif effect_type == "mysterious_cool":
                for i, glow_color in enumerate(glow_colors):
                    result = self._apply_cool_glow_from_preset(
                        result, glow_color, bg_highlights, final_glow_application_mask, scale_factor, i
                    )
                    
            elif effect_type == "cyberpunk_complex" and complex_glows:
                h, w = image_bchw.shape[-2:]
                device = image_bchw.device
                atmospheric_depth = self._create_atmospheric_depth_map(
                    image_analysis, mask_system, scale_factor, device, h, w
                )
                result = self._apply_cyberpunk_atmospheric_effects(
                    result, glow_colors, atmospheric_depth, {'background_mask': final_glow_application_mask}, scale_factor, image_analysis,
                    intensity_multiplier, saturation_boost, coverage_multiplier
                )
            
            result = self.apply_clipping_protection(result)
            return result

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in preset atmospheric effects: {e}")

            return image_bchw
    
    def _apply_natural_glow_from_preset(self, image_bchw, glow_color, bg_highlights, background_mask, scale_factor, layer_index):
  
        try:
            check_for_interruption()
            
            device = image_bchw.device
            
            glow_radius = max(40.0 * scale_factor, 20.0) * (1.0 + layer_index * 0.2)
            kernel_size = int(glow_radius * 2) | 1
            
            natural_glow = kornia.filters.gaussian_blur2d(
                bg_highlights, (kernel_size, kernel_size), (glow_radius, glow_radius)
            )
            
            glow_strength = 0.35 * (0.8 ** layer_index)
            glow_color_tensor = torch.tensor(glow_color, device=device).view(1, 3, 1, 1)
            
            glow_effect = natural_glow * glow_color_tensor * glow_strength
            result = image_bchw + glow_effect * background_mask
            
            return torch.clamp(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in natural glow from preset: {e}")
            return image_bchw

    def _apply_cinematic_glow_from_preset(self, image_bchw, glow_color, bg_highlights, background_mask, scale_factor, layer_index):
 
        try:
            check_for_interruption()
            
            device = image_bchw.device
            
            glow_radius = max(30.0 * scale_factor, 15.0) * (1.0 + layer_index * 0.3)
            kernel_size = int(glow_radius * 2) | 1
            
            cinematic_glow = kornia.filters.gaussian_blur2d(
                bg_highlights, (kernel_size, kernel_size), (glow_radius, glow_radius)
            )
            
            glow_strength = 0.4 * (0.7 ** layer_index)
            glow_color_tensor = torch.tensor(glow_color, device=device).view(1, 3, 1, 1)
            
            glow_layer = cinematic_glow * glow_color_tensor * glow_strength
            glowed_image = 1.0 - (1.0 - image_bchw) * (1.0 - glow_layer)
            
            final_mask = cinematic_glow * glow_strength * background_mask
            result = torch.lerp(image_bchw, glowed_image, final_mask)
            
            return torch.clamp(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in cinematic glow from preset: {e}")
            return image_bchw

    def _apply_cool_glow_from_preset(self, image_bchw, glow_color, bg_highlights, background_mask, scale_factor, layer_index):
   
        try:
            check_for_interruption()
            
            device = image_bchw.device
            
            glow_radius = max(25.0 * scale_factor, 12.0) * (1.0 + layer_index * 0.3)
            kernel_size = int(glow_radius * 2) | 1
            
            gray = kornia.color.rgb_to_grayscale(image_bchw)
            edges = kornia.filters.sobel(gray)
            edge_enhanced_highlights = bg_highlights * (0.8 + 0.2 * torch.tanh(edges * 2.0))
            
            cool_glow = kornia.filters.gaussian_blur2d(
                edge_enhanced_highlights, (kernel_size, kernel_size), (glow_radius, glow_radius)
            )
            
            glow_strength = 0.35 * (0.75 ** layer_index)
            glow_color_tensor = torch.tensor(glow_color, device=device).view(1, 3, 1, 1)
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            glow_lab = kornia.color.rgb_to_lab(glow_color_tensor.expand_as(image_bchw))
            
            glow_mask = cool_glow * glow_strength * background_mask
            enhanced_lab = torch.cat([
                image_lab[:, 0:1],
                torch.lerp(image_lab[:, 1:2], glow_lab[:, 1:2], glow_mask * 0.6),
                torch.lerp(image_lab[:, 2:3], glow_lab[:, 2:3], glow_mask * 0.6)
            ], dim=1)
            
            result = kornia.color.lab_to_rgb(enhanced_lab)
            return torch.clamp(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in cool glow from preset: {e}")
            return image_bchw

    def _apply_cyberpunk_atmospheric_effects(self, image_bchw, glow_colors, atmospheric_depth, mask_system, scale_factor, image_analysis, intensity_multiplier=1.0, saturation_boost=1.0, coverage_multiplier=1.0):
      
        try:
            check_for_interruption()
            
            device = image_bchw.device
            result = image_bchw.clone()
            background_mask = mask_system['background_mask']
            
            for i, glow_color in enumerate(glow_colors):
                
                base_radius = max(15.0 * scale_factor, 8.0)
                layer_radius = base_radius * (0.8 + i * 0.3)
                
                h, w = atmospheric_depth.shape[-2:]
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(0, 1, h, device=device),
                    torch.linspace(0, 1, w, device=device),
                    indexing='ij'
                )
                
                intensity_variation = (
                    torch.sin(x_coords * 3.14159 * 2) * 0.3 + 
                    torch.cos(y_coords * 3.14159 * 1.5) * 0.2 + 
                    0.5
                ).unsqueeze(0).unsqueeze(0)
                
                cyberpunk_depth = atmospheric_depth * intensity_variation
                
                kernel_size = int(layer_radius * 2) | 1
                cyberpunk_glow = kornia.filters.gaussian_blur2d(
                    cyberpunk_depth, (kernel_size, kernel_size), (layer_radius, layer_radius)
                )
                
                base_glow_strength = 0.4 * (0.6 ** i) * intensity_multiplier
                if image_analysis['is_dark_image']:
                    base_glow_strength *= 1.4

                glow_color_tensor = torch.tensor(glow_color, device=device).view(1, 3, 1, 1)

                if saturation_boost > 1.0:
                    glow_color_hsv = self._rgb_to_hsv_simple(glow_color_tensor)
                    glow_color_hsv[:, 1:2] = torch.clamp(glow_color_hsv[:, 1:2] * saturation_boost, 0.0, 1.0)
                    glow_color_tensor = self._hsv_to_rgb_simple(glow_color_hsv)

                glow_layer = cyberpunk_glow * glow_color_tensor * base_glow_strength
                glowed_image = 1.0 - (1.0 - result) * (1.0 - glow_layer)

                expanded_glow = torch.clamp(cyberpunk_glow * coverage_multiplier, 0.0, 1.0)
                glow_mask = torch.clamp(expanded_glow * base_glow_strength, 0.0, 1.0) * background_mask
                result = torch.lerp(result, glowed_image, glow_mask)
            
            result = self._add_cyberpunk_edge_effects(result, mask_system, scale_factor, glow_colors[0] if glow_colors else [1.0, 0.2, 0.8])
            
            return torch.clamp(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in cyberpunk atmospheric effects: {e}")
            return image_bchw

    def _add_cyberpunk_edge_effects(self, image_bchw, mask_system, scale_factor, primary_color):
    
        try:
            check_for_interruption()
            
            device = image_bchw.device
            background_mask = mask_system['background_mask']
            
            gray = kornia.color.rgb_to_grayscale(image_bchw)
            edges = kornia.filters.sobel(gray)
            
            edge_radius = max(3.0 * scale_factor, 2.0)
            kernel_size = int(edge_radius * 2) | 1
            soft_edges = kornia.filters.gaussian_blur2d(
                edges, (kernel_size, kernel_size), (edge_radius, edge_radius)
            )
            
            edge_strength = 0.15
            edge_color_tensor = torch.tensor(primary_color, device=device).view(1, 3, 1, 1)
            
            edge_glow = soft_edges * edge_color_tensor * edge_strength * background_mask
            edge_enhanced = 1.0 - (1.0 - image_bchw) * (1.0 - edge_glow)
            
            edge_mask = torch.clamp(soft_edges * edge_strength, 0.0, 0.3) * background_mask
            result = torch.lerp(image_bchw, edge_enhanced, edge_mask)
            
            return result
            
        except Exception as e:
            print(f"Error adding cyberpunk edge effects: {e}")
            return image_bchw
    
    def _apply_preset_color_grading(self, image_bchw, color_palette, image_analysis, mask_system):
   
        try:
            check_for_interruption()
            
            background_mask = mask_system['background_mask']
            
            highlight_tint = color_palette.get("highlight_tint")
            shadow_tint = color_palette.get("shadow_tint")
            accent_colors = color_palette.get("accent_colors", [])
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            l_channel = image_lab[:, 0:1]
            l_normalized = l_channel / 100.0
            
            tint_strength = 0.8 if image_analysis.get('is_high_contrast', False) else 1.0
            
            result = image_bchw.clone()

            if highlight_tint:
                highlight_regions = torch.pow(l_normalized, 0.6)
                highlight_mask = highlight_regions * background_mask * tint_strength * 0.7
                result = self._tint_in_lab(result, highlight_tint, highlight_mask)

            if shadow_tint:
                shadow_regions = torch.pow(1.0 - l_normalized, 1.0)
                shadow_mask = shadow_regions * background_mask * tint_strength * 0.6
                result = self._tint_in_lab(result, shadow_tint, shadow_mask)

            mood_type = image_analysis.get('mood_type', 'bright_natural')
            if mood_type in ["cyberpunk_vibrant", "warm_cinematic"] and accent_colors:
                midtone_regions = torch.exp(-((l_normalized - 0.5) / 0.3)**2)
                
                accent_mask = midtone_regions * background_mask * tint_strength * 0.3
                result = self._tint_in_lab(result, accent_colors[0], accent_mask)
            
            return torch.clamp(result, 0.0, 1.0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in preset color grading: {e}")

            return image_bchw

    def _apply_preset_temperature(self, image_bchw, temperature_shift, image_analysis, mask_system):
     
        try:
            check_for_interruption()
            background_mask = mask_system['background_mask']
            
            shift_amount = 0.0
            if isinstance(temperature_shift, str):
                shift_mapping = {"slightly_cooler": -15.0, "much_cooler": -30.0, "cooler": -22.0, 
                                    "slightly_warmer": 15.0, "much_warmer": 30.0, "warmer": 22.0}
                shift_amount = shift_mapping.get(temperature_shift, 0.0)
            elif isinstance(temperature_shift, (int, float)):
                shift_amount = temperature_shift

            if shift_amount == 0.0:
                return image_bchw
            
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            b_channel = image_lab[:, 2:3]
            b_adjusted = b_channel + (shift_amount * background_mask)
            shifted_lab = torch.cat([image_lab[:, 0:2], b_adjusted], dim=1)
            shifted_rgb = kornia.color.lab_to_rgb(shifted_lab)
            return torch.clamp(shifted_rgb, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in preset temperature: {e}")
            return image_bchw
   
    def _create_atmospheric_depth_map(self, image_analysis, mask_system, scale_factor, device, h, w):
   
        try:
            if 'far_regions' in mask_system and torch.sum(mask_system['far_regions']) > 0:
                atmospheric_depth = mask_system['far_regions'].clone()
            else:
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(0, 1, h, device=device),
                    torch.linspace(0, 1, w, device=device),
                    indexing='ij'
                )
                
                center_distance = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
                atmospheric_depth = torch.clamp(center_distance * 1.5, 0, 1).unsqueeze(0).unsqueeze(0)
            
            brightness_map = image_analysis['highlight_mask'] * 0.3 + image_analysis['shadow_mask'] * 0.7
            atmospheric_depth = atmospheric_depth * (0.7 + 0.3 * brightness_map)
            
            atmospheric_depth = atmospheric_depth * mask_system['background_mask']
            
            return atmospheric_depth
            
        except Exception as e:
            print(f"Error creating atmospheric depth map: {e}")
            return torch.zeros((1, 1, h, w), device=device)
    
    def _rgb_to_hsv_simple(self, rgb_tensor):
     
        try:
            r, g, b = rgb_tensor[:, 0:1], rgb_tensor[:, 1:2], rgb_tensor[:, 2:3]
            
            max_val = torch.max(torch.max(r, g), b)
            min_val = torch.min(torch.min(r, g), b)
            diff = max_val - min_val
            
            v = max_val
            
            s = torch.where(max_val > 1e-8, diff / max_val, torch.zeros_like(max_val))
            
            h = torch.zeros_like(diff)
            red_mask = (max_val == r) & (diff > 1e-8)
            h = torch.where(red_mask, (g - b) / (diff + 1e-8), h)
            green_mask = (max_val == g) & (diff > 1e-8)
            h = torch.where(green_mask, 2.0 + (b - r) / (diff + 1e-8), h)
            blue_mask = (max_val == b) & (diff > 1e-8)
            h = torch.where(blue_mask, 4.0 + (r - g) / (diff + 1e-8), h)
            
            h = h / 6.0
            h = torch.where(h < 0, h + 1.0, h)
            
            return torch.cat([h, s, v], dim=1)
        except Exception as e:
            print(f"Error in RGB to HSV: {e}")
            return rgb_tensor

    def _hsv_to_rgb_simple(self, hsv_tensor):
       
        try:
            h, s, v = hsv_tensor[:, 0:1], hsv_tensor[:, 1:2], hsv_tensor[:, 2:3]
            
            c = v * s
            h_prime = h * 6.0
            x = c * (1.0 - torch.abs((h_prime % 2.0) - 1.0))
            m = v - c
            
            r = torch.zeros_like(h)
            g = torch.zeros_like(h)
            b = torch.zeros_like(h)
            
            mask0 = (h_prime >= 0) & (h_prime < 1)
            r = torch.where(mask0, c, r)
            g = torch.where(mask0, x, g)
            
            mask1 = (h_prime >= 1) & (h_prime < 2)
            r = torch.where(mask1, x, r)
            g = torch.where(mask1, c, g)
            
            mask2 = (h_prime >= 2) & (h_prime < 3)
            g = torch.where(mask2, c, g)
            b = torch.where(mask2, x, b)
            
            mask3 = (h_prime >= 3) & (h_prime < 4)
            g = torch.where(mask3, x, g)
            b = torch.where(mask3, c, b)
            
            mask4 = (h_prime >= 4) & (h_prime < 5)
            r = torch.where(mask4, x, r)
            b = torch.where(mask4, c, b)
            
            mask5 = (h_prime >= 5) & (h_prime < 6)
            r = torch.where(mask5, c, r)
            b = torch.where(mask5, x, b)
            
            return torch.cat([r + m, g + m, b + m], dim=1)
        except Exception as e:
            print(f"Error in HSV to RGB: {e}")
            return hsv_tensor
    
    def get_mask_bounding_box(self, mask):
       
        if mask.sum() == 0:
            return 0, 0, mask.shape[3], mask.shape[2]
        
        rows = torch.any(mask, dim=3).squeeze()
        cols = torch.any(mask, dim=2).squeeze()
        
        y_min, y_max = torch.where(rows)[0][[0, -1]]
        x_min, x_max = torch.where(cols)[0][[0, -1]]
        
        return x_min.item(), y_min.item(), x_max.item(), y_max.item()
   

    def _apply_dof_pyramid(self, image_bchw, subject_mask, depth_map, dof_strength, device):
  
        try:
            check_for_interruption()

            if depth_map is None or subject_mask is None:
                print("Refiner: DOF skipped, missing depth map or subject mask.")
                return image_bchw

            background_area_mask = 1.0 - subject_mask
            background_plate_with_hole = image_bchw * background_area_mask
            bleed_radius = max(5.0, dof_strength * 2.0)
            k_size_bleed = int(bleed_radius * 2) | 1
            image_for_bleed = kornia.filters.gaussian_blur2d(image_bchw, (k_size_bleed, k_size_bleed), (bleed_radius, bleed_radius))
            inpainted_background = torch.lerp(image_for_bleed, background_plate_with_hole, background_area_mask)
            
            num_levels = 4
            blur_map = (1.0 - depth_map)
            pyramid = [inpainted_background]
            for i in range(1, num_levels):
                level_strength = (dof_strength * 2.0) * (i / (num_levels - 1))
                k_size = int(level_strength * 2) | 1
                if k_size >= 3:
                    pyramid.append(kornia.filters.gaussian_blur2d(inpainted_background, (k_size, k_size), (level_strength, level_strength)))
                else:
                    pyramid.append(inpainted_background)
            
            scaled_map = blur_map * (num_levels - 1)
            final_blurred_background = pyramid[0]
            for i in range(num_levels - 1):
                final_blurred_background = torch.lerp(final_blurred_background, pyramid[i+1], torch.clamp(scaled_map - i, 0.0, 1.0))



            transition_blur_radius = max(1.5, dof_strength * 0.7)
            k_size_trans = int(transition_blur_radius * 2) | 1
            transition_blur_image = kornia.filters.gaussian_blur2d(image_bchw, (k_size_trans, k_size_trans), (transition_blur_radius, transition_blur_radius))

            core_mask = subject_mask.pow(3.0)


            image_with_transition_fg = torch.lerp(final_blurred_background, transition_blur_image, subject_mask)

            final_image = torch.lerp(image_with_transition_fg, image_bchw, core_mask)

            return final_image

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: Error in _apply_dof_pyramid (v6 w/ Explicit Layering): {e}")

            return image_bchw

        
    def apply_correction_photo(self, image_bchw, subject_mask, strength):
      
        try:
            check_for_interruption()

            if strength <= 0:
                return image_bchw

            if not hasattr(kornia.enhance, 'equalize_clahe'):
                print("-------------------------------------------------------------------------------------------------")
                print("WARNING: kornia.enhance.equalize_clahe not found.")
                print("Your version of kornia may be outdated. Please update it.")
                print("Skipping Corrective (Auto) relighting.")
                print("-------------------------------------------------------------------------------------------------")
                return image_bchw

            x_min, y_min, x_max, y_max = self.get_mask_bounding_box(subject_mask)

            padding = 20
            h, w = image_bchw.shape[-2:]
            crop_x_min = max(0, x_min - padding)
            crop_y_min = max(0, y_min - padding)
            crop_x_max = min(w, x_max + padding)
            crop_y_max = min(h, y_max + padding)

            if crop_x_min >= crop_x_max or crop_y_min >= crop_y_max:
                    return image_bchw

            subject_crop_bchw = image_bchw[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            subject_crop_lab = kornia.color.rgb_to_lab(subject_crop_bchw)

            l_channel = subject_crop_lab[:, 0:1, :, :]

            clahe_clip_limit = 2.5
            corrected_l_channel = kornia.enhance.equalize_clahe(l_channel / 100.0, clip_limit=clahe_clip_limit) * 100.0

            corrected_lab_crop = torch.cat([corrected_l_channel, subject_crop_lab[:, 1:3, :, :]], dim=1)
            
            corrected_rgb_crop = kornia.color.lab_to_rgb(corrected_lab_crop)

            corrected_image = image_bchw.clone()
            corrected_image[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max] = corrected_rgb_crop
            
            blending_strength = pow(strength, 1.5)
            
            final_image = torch.lerp(image_bchw, corrected_image, blending_strength * subject_mask)

            final_image = self.apply_clipping_protection(final_image)
            return final_image

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in automatic light correction (CLAHE): {e}")

            return image_bchw
              
   
    def apply_professional_relighting(self, image_bchw, subject_mask, strength, device):

        try:
            check_for_interruption()
            
            h, w = image_bchw.shape[-2:]
            scale_factor = min(h, w) / 1024.0
            
            mask_system = self.create_smooth_subject_mask(subject_mask, scale_factor)
            
            detail_preservation = self.create_detail_preservation_mask(image_bchw, subject_mask, scale_factor)
            
            key_light = self.create_key_light(h, w, device, scale_factor)
            fill_light = self.create_fill_light(h, w, device, scale_factor)
            
            lighting_analysis = self.analyze_subject_lighting(image_bchw, subject_mask)
            
            relit_image = self.apply_adaptive_studio_lighting_smooth(
                image_bchw, mask_system, detail_preservation, 
                key_light, fill_light,  
                lighting_analysis, strength
            )
            
            relit_image = self.apply_clipping_protection(relit_image)
            return relit_image
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in professional relighting: {e}")
            return image_bchw

    def apply_adaptive_studio_lighting_smooth(self, image_bchw, mask_system, detail_preservation,
                                            key_light, fill_light, lighting_analysis, strength):
        try:
            check_for_interruption()

            if strength <= 0:
                return image_bchw

            fully_relit_image = image_bchw.clone()

            exposure_strength = 0.5
            key_strength = 0.35 * (0.8 if lighting_analysis['avg_brightness'] > 0.7 else 1.2)
            fill_strength = 0.2 + (lighting_analysis['shadow_severity'] * 0.25)
            color_strength = 0.6
            contrast_strength = 0.45

            fully_relit_image = self.apply_smart_exposure_correction_smooth(
                fully_relit_image, mask_system, detail_preservation, exposure_strength
            )
            
            fully_relit_image = self.apply_light_to_subject_smooth(
                fully_relit_image, mask_system, detail_preservation, key_light, key_strength, 'key'
            )
            fully_relit_image = self.apply_light_to_subject_smooth(
                fully_relit_image, mask_system, detail_preservation, fill_light, fill_strength, 'fill'
            )
       

            if abs(lighting_analysis['color_temperature']) > 0.1:
                fully_relit_image = self.apply_color_temperature_correction_smooth(
                    fully_relit_image, mask_system, lighting_analysis['color_temperature'], color_strength
                )
            fully_relit_image = self.apply_subject_contrast_enhancement_smooth(
                fully_relit_image, mask_system, detail_preservation, contrast_strength
            )

            blending_strength = pow(strength, 1.5)
            relit_image = torch.lerp(image_bchw, fully_relit_image, blending_strength * mask_system['combined_mask'])
            
            if blending_strength > 0:
                gamma = 1.0 + (0.15 * blending_strength)
                darkened_image = torch.clamp(relit_image.pow(gamma), 0.0, 1.0)
                final_image = torch.lerp(relit_image, darkened_image, mask_system['combined_mask'])
            else:
                final_image = relit_image

            return final_image
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth adaptive studio lighting: {e}")
            return image_bchw

    def create_smooth_subject_mask(self, subject_mask, scale_factor):
        try:
            check_for_interruption()
            
            device = subject_mask.device
            
            edge_blur = max(3.0 * scale_factor, 2.0)
            edge_kernel_size = int(edge_blur * 2) | 1
            
            soft_mask = kornia.filters.gaussian_blur2d(
                subject_mask,
                (edge_kernel_size, edge_kernel_size),
                (edge_blur, edge_blur)
            )
            
            falloff_blur = max(15.0 * scale_factor, 8.0)
            falloff_kernel_size = int(falloff_blur * 2) | 1
            
            falloff_mask = kornia.filters.gaussian_blur2d(
                subject_mask,
                (falloff_kernel_size, falloff_kernel_size),
                (falloff_blur, falloff_blur)
            )
            
            smooth_mask = soft_mask * 0.7 + falloff_mask * 0.3
            
            edge_transition = torch.clamp(falloff_mask - soft_mask + 0.2, 0, 1)
            
            return {
                'core_mask': soft_mask,
                'falloff_mask': falloff_mask, 
                'transition_mask': edge_transition,
                'combined_mask': smooth_mask
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating smooth subject mask: {e}")
            return {
                'core_mask': subject_mask,
                'falloff_mask': subject_mask,
                'transition_mask': subject_mask,
                'combined_mask': subject_mask
            }

    def create_detail_preservation_mask(self, image_bchw, subject_mask, scale_factor):
        try:
            check_for_interruption()
            
            device = image_bchw.device
            
            gray = 0.299 * image_bchw[:, 0:1] + 0.587 * image_bchw[:, 1:2] + 0.114 * image_bchw[:, 2:3]
            
            detail_small = kornia.filters.sobel(gray)
            
            blur_medium = kornia.filters.gaussian_blur2d(gray, (5, 5), (1.5, 1.5))
            detail_medium = torch.abs(gray - blur_medium)
            
            blur_large = kornia.filters.gaussian_blur2d(gray, (9, 9), (3.0, 3.0))
            detail_large = torch.abs(gray - blur_large)
            
            combined_details = (detail_small * 0.4 + detail_medium * 0.4 + detail_large * 0.2)
            
            detail_strength = torch.tanh(combined_details * 8.0)
            
            shadow_areas = torch.exp(-(gray / 0.4)**2)
            detail_strength = detail_strength * shadow_areas
            
            smooth_radius = max(2.0 * scale_factor, 1.0)
            smooth_kernel_size = int(smooth_radius * 2) | 1
            
            detail_preservation = kornia.filters.gaussian_blur2d(
                detail_strength,
                (smooth_kernel_size, smooth_kernel_size),
                (smooth_radius, smooth_radius)
            )
            
            detail_preservation = detail_preservation * subject_mask
            
            return torch.clamp(detail_preservation, 0, 0.8)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating detail preservation mask: {e}")
            return torch.zeros_like(subject_mask)

    def create_key_light(self, h, w, device, scale_factor):
        light_x = 0.5
        light_y = 0.3
        
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        
        distance = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
        
        falloff = 1.0 / (1.0 + (distance / 0.8)**1.5)
        
        key_light = falloff
        
        blur_radius = max(25.0 * scale_factor, 8.0)
        kernel_size = int(blur_radius * 2) | 1
        key_light = kornia.filters.gaussian_blur2d(
            key_light.unsqueeze(0).unsqueeze(0),
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        ).squeeze()
        
        key_light = key_light * 0.15
        
        return key_light.unsqueeze(0).unsqueeze(0)

    def create_fill_light(self, h, w, device, scale_factor):
        light_x = 0.75
        light_y = 0.4
        
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        
        distance = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
        
        falloff = 1.0 / (1.0 + (distance / 0.6)**1.5)
        
        blur_radius = max(15.0 * scale_factor, 4.0)
        kernel_size = int(blur_radius * 2) | 1
        fill_light = kornia.filters.gaussian_blur2d(
            falloff.unsqueeze(0).unsqueeze(0),
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        ).squeeze()
        
        return fill_light.unsqueeze(0).unsqueeze(0) * 0.06

    def analyze_subject_lighting(self, image_bchw, subject_mask):
        try:
            check_for_interruption()
            
            subject_pixels = image_bchw * subject_mask
            valid_pixels = subject_pixels[subject_mask.bool().expand_as(subject_pixels)]
            
            if valid_pixels.numel() == 0:
                return {
                    'avg_brightness': 0.5,
                    'contrast_ratio': 1.0,
                    'is_underlit': False,
                    'shadow_severity': 0.0,
                    'color_temperature': 0.0
                }
            
            valid_pixels = valid_pixels.view(-1, 3)
            
            luminance = 0.299 * valid_pixels[:, 0] + 0.587 * valid_pixels[:, 1] + 0.114 * valid_pixels[:, 2]
            
            avg_brightness = torch.mean(luminance)
            brightness_std = torch.std(luminance)
            
            is_underlit = avg_brightness < 0.4
            
            shadow_pixels = (luminance < 0.2).float()
            shadow_severity = torch.mean(shadow_pixels)
            
            avg_red = torch.mean(valid_pixels[:, 0])
            avg_blue = torch.mean(valid_pixels[:, 2])
            color_temperature = (avg_red - avg_blue).clamp(-0.3, 0.3)
            
            contrast_ratio = brightness_std / (avg_brightness + 0.001)
            
            return {
                'avg_brightness': avg_brightness.item(),
                'contrast_ratio': contrast_ratio.item(),
                'is_underlit': is_underlit.item(),
                'shadow_severity': shadow_severity.item(),
                'color_temperature': color_temperature.item()
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in lighting analysis: {e}")
            return {
                'avg_brightness': 0.5,
                'contrast_ratio': 1.0,
                'is_underlit': False,
                'shadow_severity': 0.0,
                'color_temperature': 0.0
            }

    def apply_smart_exposure_correction_smooth(self, image_bchw, mask_system, detail_preservation, strength):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
                
            result = image_bchw.clone()
            
            luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            
            shadow_mask = torch.exp(-(luminance / 0.3)**2)
            midtone_mask = torch.exp(-((luminance - 0.5) / 0.3)**2)
            highlight_mask = torch.exp(-((luminance - 0.8) / 0.2)**2)
            
            detail_reduction = 1.0 - (detail_preservation * 0.6)
            
            shadow_lift = strength * 0.05 * shadow_mask * detail_reduction
            midtone_boost = strength * 0.02 * midtone_mask * detail_reduction
            highlight_protection = -strength * 0.01 * highlight_mask
            
            core_adjustment = (shadow_lift + midtone_boost + highlight_protection) * mask_system['core_mask']
            falloff_adjustment = (shadow_lift + midtone_boost + highlight_protection) * 0.6 * mask_system['falloff_mask']
            transition_adjustment = (shadow_lift + midtone_boost + highlight_protection) * 0.3 * mask_system['transition_mask']
            
            total_adjustment = core_adjustment + falloff_adjustment + transition_adjustment
            
            total_adjustment = torch.clamp(total_adjustment, -0.1, 0.1)
            
            for c in range(3):
                channel = result[:, c:c+1, :, :]
                adjusted = channel + total_adjustment
                result[:, c:c+1, :, :] = torch.clamp(adjusted, 0, 1)
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth smart exposure correction: {e}")
            return image_bchw

    def apply_subject_contrast_enhancement_smooth(self, image_bchw, mask_system, detail_preservation, strength):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
                
            result = image_bchw.clone()
            
            luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            
            detail_reduction = 1.0 - (detail_preservation * 0.8)
            
            enhanced_luminance = luminance + torch.sin(luminance * 3.14159) * strength * 0.1 * detail_reduction
            enhanced_luminance = torch.clamp(enhanced_luminance, 0, 1)
            
            luminance_ratio = enhanced_luminance / (luminance + 1e-8)
            luminance_ratio = torch.clamp(luminance_ratio, 0.7, 1.5)
            
            core_ratio = torch.lerp(torch.ones_like(luminance_ratio), luminance_ratio, mask_system['core_mask'])
            falloff_ratio = torch.lerp(torch.ones_like(luminance_ratio), luminance_ratio, mask_system['falloff_mask'] * 0.6)
            transition_ratio = torch.lerp(torch.ones_like(luminance_ratio), luminance_ratio, mask_system['transition_mask'] * 0.3)
            
            final_ratio = core_ratio * falloff_ratio * transition_ratio
            
            for c in range(3):
                channel = result[:, c:c+1, :, :]
                enhanced_channel = channel * final_ratio
                result[:, c:c+1, :, :] = torch.clamp(enhanced_channel, 0, 1)
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth subject contrast enhancement: {e}")
            return image_bchw

    def apply_light_to_subject_smooth(self, image_bchw, mask_system, detail_preservation, light_map, strength, light_type):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
            
            device = image_bchw.device
            
            if light_map.shape[-2:] != image_bchw.shape[-2:]:
                light_map = F.interpolate(light_map, size=image_bchw.shape[-2:], mode='bilinear', align_corners=False)
            
            result = image_bchw.clone()
            
            if light_type == 'key':
                warm_r, warm_g, warm_b = 1.0, 0.98, 0.95
                
                luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
                shadow_mask = torch.exp(-(luminance / 0.4)**2)
                
                detail_reduction = 1.0 - (detail_preservation * 0.3)
                
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    light_color = [warm_r, warm_g, warm_b][c]
                    
                    core_strength = strength * shadow_mask * detail_reduction * mask_system['core_mask']
                    falloff_strength = strength * 0.4 * shadow_mask * mask_system['falloff_mask']
                    transition_strength = strength * 0.2 * shadow_mask * mask_system['transition_mask']
                    
                    total_strength = torch.clamp(core_strength + falloff_strength + transition_strength, 0, strength)
                    
                    light_effect = light_map * total_strength * light_color * 0.8 
                    blended_channel = 1.0 - (1.0 - channel) * (1.0 - light_effect)
                    result[:, c:c+1, :, :] = torch.clamp(blended_channel, 0, 1)

            elif light_type == 'fill':
                luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
                shadow_mask = torch.exp(-(luminance / 0.25)**2)
                
                detail_reduction = 1.0 - (detail_preservation * 0.5)
                fill_colors = [0.98, 1.0, 1.02]
                
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    fill_color = fill_colors[c]
                    
                    core_fill = strength * 0.3 * shadow_mask * detail_reduction * mask_system['core_mask']
                    falloff_fill = strength * 0.2 * shadow_mask * mask_system['falloff_mask']
                    
                    combined_fill_effect = light_map * (core_fill + falloff_fill) * fill_color
                    lifted = 1.0 - (1.0 - channel) * (1.0 - combined_fill_effect)
                    result[:, c:c+1, :, :] = torch.clamp(lifted, 0, 1)
            
            elif light_type == 'rim':
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    
                    rim_effect = light_map * strength * 0.6 * mask_system['core_mask']
                    
                    highlighted = 1.0 - (1.0 - channel) * (1.0 - rim_effect)
                    
                    rim_lerp_mask = torch.clamp(rim_effect * 10.0, 0, 1)
                    result[:, c:c+1, :, :] = torch.lerp(channel, highlighted, rim_lerp_mask)
            
            return torch.clamp(result, 0, 1)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying {light_type} light with smooth blending: {e}")
            return image_bchw

    def apply_color_temperature_correction_smooth(self, image_bchw, mask_system, temperature_bias, strength):
        try:
            check_for_interruption()
            
            if abs(temperature_bias) < 0.05 or strength <= 0:
                return image_bchw
            
            result = image_bchw.clone()
            
            if temperature_bias > 0:
                red_adjust = -temperature_bias * strength
                blue_adjust = temperature_bias * strength * 0.8
            else:
                red_adjust = -temperature_bias * strength * 0.8
                blue_adjust = temperature_bias * strength
            
            core_strength = mask_system['core_mask']
            falloff_strength = mask_system['falloff_mask'] * 0.6
            transition_strength = mask_system['transition_mask'] * 0.3
            
            total_mask = core_strength + falloff_strength + transition_strength
            
            result[:, 0:1, :, :] = torch.clamp(
                result[:, 0:1, :, :] + red_adjust * total_mask, 0, 1
            )
            
            result[:, 2:3, :, :] = torch.clamp(
                result[:, 2:3, :, :] + blue_adjust * total_mask, 0, 1
            )
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth color temperature correction: {e}")
            return image_bchw
    
    def apply_dynamic_contrast_gpu(self, image_tensor, strength):

        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_dynamic_contrast_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            l_channel = image_lab[:, 0:1, :, :]
            l_normalized = l_channel / 100.0
            
            enhanced_l_normalized = self.apply_simple_s_curve(l_normalized, strength)
            
            enhanced_l = enhanced_l_normalized * 100.0
            
            enhanced_lab = torch.cat([enhanced_l, image_lab[:, 1:3, :, :]], dim=1)
            
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during dynamic contrast enhancement: {e}. Returning original tensor.")
            return image_tensor

    def apply_simple_s_curve(self, luminance, strength):
 
        if strength == 0.0:
            return luminance
        
        image_stats = self.analyze_image_dynamics(luminance)
        
        shadow_adjustment = self.calculate_shadow_adjustment(luminance, image_stats, strength)
        highlight_adjustment = self.calculate_highlight_adjustment(luminance, image_stats, strength)
        midtone_adjustment = self.calculate_midtone_adjustment(luminance, image_stats, strength)
        
        enhanced = luminance + shadow_adjustment + highlight_adjustment + midtone_adjustment
        
        return torch.clamp(enhanced, 0.0, 1.0)
    def analyze_image_dynamics(self, luminance):
  
        mean_luminance = torch.mean(luminance)
        std_luminance = torch.std(luminance)
        
        shadow_amount = torch.mean((luminance < 0.25).float())
        highlight_amount = torch.mean((luminance > 0.75).float())
        midtone_amount = torch.mean(torch.logical_and(luminance >= 0.25, luminance <= 0.75).float())
        
        contrast_level = std_luminance / (mean_luminance + 0.001)
        
        is_low_contrast = contrast_level < 0.3
        
        is_high_contrast = contrast_level > 0.6
        
        is_underexposed = mean_luminance < 0.4
        is_overexposed = mean_luminance > 0.7
        
        return {
            'mean_luminance': mean_luminance,
            'contrast_level': contrast_level,
            'shadow_amount': shadow_amount,
            'highlight_amount': highlight_amount,
            'midtone_amount': midtone_amount,
            'is_low_contrast': is_low_contrast,
            'is_high_contrast': is_high_contrast,
            'is_underexposed': is_underexposed,
            'is_overexposed': is_overexposed
        }
    def calculate_shadow_adjustment(self, luminance, stats, strength):
 
        base_shadow_strength = strength * 0.18
        
        if stats['is_underexposed']:
            shadow_strength = base_shadow_strength * 0.5
        elif stats['shadow_amount'] < 0.1:
            shadow_strength = base_shadow_strength * 1.5
        elif stats['is_low_contrast']:
            shadow_strength = base_shadow_strength * 1.3
        else:
            shadow_strength = base_shadow_strength
        
        shadows_mask = torch.exp(-((luminance - 0.0) / 0.25) ** 2)
        
        shadow_adjustment = -shadow_strength * shadows_mask * (0.25 - luminance).clamp(0, 1)
        
        return shadow_adjustment
    def calculate_highlight_adjustment(self, luminance, stats, strength):
      
        base_highlight_strength = strength * 0.12
        
        if stats['is_overexposed']:
            highlight_strength = base_highlight_strength * 2.0
        elif stats['highlight_amount'] > 0.2:
            highlight_strength = base_highlight_strength * 1.4
        elif stats['highlight_amount'] < 0.05:
            highlight_strength = base_highlight_strength * 0.6
        else:
            highlight_strength = base_highlight_strength
        
        highlights_mask = torch.exp(-((luminance - 1.0) / 0.25) ** 2)
        
        highlight_adjustment = -highlight_strength * highlights_mask * (luminance - 0.75).clamp(0, 1)
        
        return highlight_adjustment
    def calculate_midtone_adjustment(self, luminance, stats, strength):
    
        base_midtone_strength = strength * 0.22
        
        if stats['is_low_contrast']:
            midtone_strength = base_midtone_strength * 1.5
        elif stats['is_high_contrast']:
            midtone_strength = base_midtone_strength * 0.7
        else:
            midtone_strength = base_midtone_strength
        
        midtones_mask = torch.exp(-((luminance - 0.5) / 0.35) ** 2)
        
        midtone_contrast_curve = torch.tanh((luminance - 0.5) * 3.0) * 0.5 + 0.5
        midtone_adjustment = midtone_strength * midtones_mask * (midtone_contrast_curve - luminance)
        
        return midtone_adjustment
 
    
    def apply_vibrance_gpu(self, image_tensor, strength):

        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_vibrance_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            a_channel = image_lab[:, 1:2, :, :]
            b_channel = image_lab[:, 2:3, :, :]
            
            current_saturation = torch.sqrt(a_channel**2 + b_channel**2)
            max_saturation = current_saturation.max()
            
            if max_saturation > 0:
                normalized_saturation = current_saturation / max_saturation
            else:
                normalized_saturation = current_saturation
            
            vibrance_mask = 1.0 - normalized_saturation
            vibrance_mask = torch.pow(vibrance_mask, 0.5)
            
            skin_protection = self.create_skin_tone_mask(a_channel, b_channel)
            final_mask = vibrance_mask * (1.0 - skin_protection * 0.7)
            
            vibrance_factor = 1.0 + (strength * 0.6) * final_mask
            
            enhanced_a = a_channel * vibrance_factor
            enhanced_b = b_channel * vibrance_factor
            
            enhanced_lab = torch.cat([image_lab[:, 0:1, :, :], enhanced_a, enhanced_b], dim=1)
            
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during vibrance enhancement: {e}. Returning original tensor.")
            return image_tensor

    def create_skin_tone_mask(self, a_channel, b_channel):

        skin_a_min, skin_a_max = 0.0, 25.0
        skin_b_min, skin_b_max = 5.0, 35.0
        
        a_in_range = torch.logical_and(a_channel >= skin_a_min, a_channel <= skin_a_max)
        b_in_range = torch.logical_and(b_channel >= skin_b_min, b_channel <= skin_b_max)
        
        skin_mask = torch.logical_and(a_in_range, b_in_range).float()
        
        kernel_size = 5
        skin_mask_blurred = F.avg_pool2d(skin_mask, kernel_size, stride=1, padding=kernel_size//2)
        
        return skin_mask_blurred
    
    def apply_clarity_gpu(self, image_tensor, strength):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_clarity_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            l_channel = image_lab[:, 0:1, :, :]
            l_normalized = l_channel / 100.0
            
            clarity_detail = self.extract_advanced_clarity_detail(l_normalized, strength)
            
            midtone_mask = self.create_midtone_mask(l_normalized)
            
            focus_mask = torch.ones_like(l_normalized)
            if self.cached_depth_map is not None:
                depth_map_device = self.cached_depth_map.to(l_normalized.device)
                if depth_map_device.shape[-2:] == l_normalized.shape[-2:]:
                    focus_mask = depth_map_device.pow(0.75)
                else:
                    print(f"Warning: Depth map dimensions ({self.cached_depth_map.shape[-2:]}) do not match image ({l_normalized.shape[-2:]}). Skipping depth awareness for clarity.")
            
            adaptive_strength = strength * midtone_mask * focus_mask
            
            enhanced_l_normalized = l_normalized + clarity_detail * adaptive_strength
            enhanced_l_normalized = torch.clamp(enhanced_l_normalized, 0.0, 1.0)
            
            enhanced_l = enhanced_l_normalized * 100.0
            enhanced_lab = torch.cat([enhanced_l, image_lab[:, 1:3, :, :]], dim=1)
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during clarity enhancement: {e}. Returning original tensor.")
            return image_tensor

    def extract_advanced_clarity_detail(self, luminance, strength):
        try:
            check_for_interruption()
            
            device = luminance.device
            dtype = luminance.dtype
            
            base_radius = max(8.0, min(luminance.shape[-1], luminance.shape[-2]) * 0.015)
            base_radius = min(base_radius, 20.0)
            
            if strength > 0.7:
                primary_radius = base_radius * 1.0
                secondary_radius = base_radius * 1.8
                primary_weight = 0.65
                secondary_weight = 0.35
            else:
                primary_radius = base_radius * 1.2
                secondary_radius = None
                primary_weight = 1.0
                secondary_weight = 0.0
            
            primary_blurred = self.advanced_gaussian_blur(luminance, primary_radius, device, dtype)
            primary_detail = (luminance - primary_blurred) * primary_weight
            
            if secondary_radius is not None:
                secondary_blurred = self.advanced_gaussian_blur(luminance, secondary_radius, device, dtype)
                secondary_detail = (luminance - secondary_blurred) * secondary_weight
                combined_detail = primary_detail + secondary_detail
            else:
                combined_detail = primary_detail
            
            edge_mask = self.create_optimized_edge_mask(luminance, device, dtype)
            combined_detail = combined_detail * edge_mask
            
            return combined_detail
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in extract_advanced_clarity_detail: {e}")
            return torch.zeros_like(luminance)

    def create_midtone_mask(self, luminance):
        try:
            check_for_interruption()
            
            distance_from_midtone = torch.abs(luminance - 0.5)
            
            midtone_mask = torch.exp(-distance_from_midtone.pow(2) * 11.111)
            
            midtone_mask = torch.clamp(midtone_mask, 0.2, 1.0)
            
            return midtone_mask
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_midtone_mask: {e}")
            return torch.ones_like(luminance)
    def create_optimized_edge_mask(self, luminance, device, dtype):
        try:
            check_for_interruption()
            
            edge_magnitude = kornia.filters.sobel(luminance)
            
            edge_mask = torch.tanh(edge_magnitude * 6.0)
            edge_mask = torch.clamp(edge_mask, 0.25, 1.0)
            
            return edge_mask
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_optimized_edge_mask: {e}")
            return torch.ones_like(luminance)

    def advanced_gaussian_blur(self, tensor, radius, device, dtype):
        try:
            check_for_interruption()
            
            kernel_size = max(3, int(radius * 2) | 1)
            kernel_size = min(kernel_size, 31)
            
            sigma = radius / 2.0
            
            tensor_bchw = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
            
            blurred = kornia.filters.gaussian_blur2d(
                tensor_bchw, 
                kernel_size=(kernel_size, kernel_size), 
                sigma=(sigma, sigma)
            )
            
            return blurred if tensor.dim() == 4 else blurred.squeeze(0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in advanced_gaussian_blur: {e}")
            return tensor
    
    def apply_smart_sharpen(self, image_tensor, strength):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_smart_sharpen expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            device = image_tensor.device
            dtype = image_tensor.dtype
            image_bchw = image_tensor.permute(0, 3, 1, 2)
            
            batch_size, channels, height, width = image_bchw.shape
            edge_fade = self.create_edge_fade_mask(height, width, device, dtype)
            
            gray = 0.299 * image_bchw[:, 0:1] + 0.587 * image_bchw[:, 1:2] + 0.114 * image_bchw[:, 2:3]
            
            blur_small = self.advanced_gaussian_blur(gray, 0.5, device, dtype)
            blur_medium = self.advanced_gaussian_blur(gray, 1.5, device, dtype)
            detail_fine = gray - blur_small
            detail_medium = gray - blur_medium
            
            detail_magnitude = torch.abs(detail_medium)
            edge_mask = torch.tanh(detail_magnitude * 20.0)
            edge_mask = edge_mask * edge_fade
            smooth_mask = torch.tanh(detail_magnitude * 5.0)
            edge_mask = edge_mask * (0.3 + 0.7 * smooth_mask)
            
            focus_mask = torch.ones_like(edge_mask)
            if self.cached_depth_map is not None:
                depth_map_device = self.cached_depth_map.to(edge_mask.device)
                h_current, w_current = edge_mask.shape[-2:]
                
                if depth_map_device.shape[-2:] != (h_current, w_current):
                    depth_map_resized_local = F.interpolate(depth_map_device, size=(h_current, w_current), mode='bilinear', align_corners=False)
                else:
                    depth_map_resized_local = depth_map_device

                focus_mask = depth_map_resized_local.pow(1.5)
            
            edge_mask = edge_mask * focus_mask

            if strength <= 1.0:
                sharpening_amount = strength * 2.0
            else:
                sharpening_amount = 2.0 + (strength - 1.0) * 8.0
            
            for c in range(3):
                channel = image_bchw[:, c:c+1]
                channel_blur = self.advanced_gaussian_blur(channel, 1.0, device, dtype)
                channel_detail = channel - channel_blur
                
                sharpening = channel_detail * edge_mask * sharpening_amount
                image_bchw[:, c:c+1] = channel + sharpening
            
            return torch.clamp(image_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smart sharpen: {e}")
            return image_tensor
  
  
    def create_edge_fade_mask(self, height, width, device, dtype):
   
        size_factor = min(height, width) / 512.0
        fade_percentage = 0.02 + (0.01 * min(1.0, size_factor / 4.0))
        fade_distance = min(height, width) * fade_percentage
        
        y_profile = torch.ones(height, device=device, dtype=dtype)
        x_profile = torch.ones(width, device=device, dtype=dtype)
        
        for i in range(int(fade_distance)):
            fade_value = i / fade_distance
            y_profile[i] = fade_value
            y_profile[-(i+1)] = fade_value
            x_profile[i] = fade_value
            x_profile[-(i+1)] = fade_value
        
        mask_2d = y_profile.unsqueeze(1) * x_profile.unsqueeze(0)
        
        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
        
        return mask_4d
