import torch
import torch.nn.functional as F
import numpy as np
import random 
import nodes
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import comfy.samplers
import comfy.sample
import latent_preview
import cv2
import kornia
import re
from .face_detector import ForbiddenVisionFaceDetector
from .mask_processor import ForbiddenVisionMaskProcessor
from .utils import check_for_interruption, get_yolo_models, get_sam_models, get_yolo_seg_models_only, get_ordered_upscaler_model_list, ensure_model_directories, load_model_preferences, clean_model_name

class ExclusionProcessor:
    def process(self, text, exclusions):
        if not exclusions or not exclusions.strip(): return text
        exclusion_list = [word.strip() for word in exclusions.split(',') if word.strip()]
        if not exclusion_list: return text
        pattern = r'\b(' + '|'.join(re.escape(word) for word in exclusion_list) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r',,+', ',', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(,\s*)+', ', ', text)
        text = text.strip(' ,')
        return text
class ForbiddenVisionFaceProcessorIntegrated:

    SDXL_RESOLUTIONS = {
        "640x1536": (640, 1536),
        "768x1344": (768, 1344),
        "832x1216": (832, 1216),
        "896x1152": (896, 1152),
        "1024x1024": (1024, 1024),
        "1152x896": (1152, 896),
        "1216x832": (1216, 832),
        "1344x768": (1344, 768),
        "1536x640": (1536, 640),
    }

    @classmethod
    def INPUT_TYPES(s):
                
        config = load_model_preferences()
        
        yolo_models = get_yolo_models()
        sam_models = get_sam_models()
        yolo_seg_models_only = get_yolo_seg_models_only()
        upscaler_models = get_ordered_upscaler_model_list()
        
        def get_preferred_default(model_list, preferred_list, fallback=None):
            clean_model_list = [clean_model_name(model) for model in model_list]
            
            for preferred in preferred_list:
                if preferred in clean_model_list:
                    original_index = clean_model_list.index(preferred)
                    return model_list[original_index]
            
            if fallback and fallback in model_list:
                return fallback
            return model_list[0] if model_list else "None Found"
        
        default_upscaler = get_preferred_default(
            upscaler_models, 
            config['preferred_models']['upscaler'], 
            "Fast 4x (Lanczos)"
        )
        
        default_bbox = get_preferred_default(
            yolo_models, 
            config['preferred_models']['bbox']
        )
        
        default_bbox_b = get_preferred_default(
            ["None"] + yolo_models, 
            config['preferred_models']['bbox'], 
            "None"
        )
        
        default_seg = get_preferred_default(
            sam_models, 
            config['preferred_models']['segmentation'], 
            "None"
        )
        
        default_seg_b = get_preferred_default(
            yolo_seg_models_only, 
            config['preferred_models']['segmentation'], 
            "None"
        )
        
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "denoise_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                "face_selection": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "0=All faces, 1=1st face, etc. Used by both mask input and internal detector."}),
                "processing_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "The resolution for processing. If adaptive resolution is on, this acts as a target area."}),
                "use_adaptive_resolution": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Enable to automatically select the best SDXL resolution based on face aspect ratio."}),
                "enable_pre_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Enable to upscale small faces with an AI model before processing."}),
                "upscaler_model": (upscaler_models, {"default": default_upscaler, "tooltip": "The model used for pre-upscaling small faces. Only active if 'enable_pre_upscale' is on."}),
                "crop_padding": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Padding added to the mask's bounding box before inpaint."}),
                
                "face_positive_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Prepend the input positive prompt with new tags"}),
                "replace_positive_prompt": ("BOOLEAN", {"default": False}),
                "face_negative_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Prepend the input negative prompt with new tags"}),
                "replace_negative_prompt": ("BOOLEAN", {"default": False}),   
                "exclusions": ("STRING", {"multiline": True, "default": "", "placeholder": "e.g. glasses, scar, wrinkles\nTags to remove from main prompt for face generation...", "tooltip": "Words/tags to remove from the main prompt specifically for the face processing step."}),

                "blend_softness": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "sampling_mask_blur_size": ("INT", {"default": 21, "min": 1, "max": 101, "step": 2}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.1}),
                "enable_vertical_flip": ("BOOLEAN", {"default": False}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                
                "bbox_model": (yolo_models, {"default": default_bbox, "tooltip": "Internal BBOX detector model (YOLO). Used if 'mask' is not connected."}),
                "bbox_model_B": (["None"] + yolo_models, {"default": default_bbox_b, "tooltip": "Optional second BBOX detector (YOLO). Runs with the first for more robust detection."}),
                "yolo_seg_model": (sam_models, {"default": default_seg, "tooltip": "Primary masking model. Includes YOLO-seg (fast, 6MB) and SAM models (slower, robust). Used if 'mask' is not connected."}),
                "yolo_seg_model_B": (yolo_seg_models_only, {"default": default_seg_b, "tooltip": "Optional second YOLO-seg model for ensemble masking. Only YOLO-seg models available."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image input. If latent is also provided, latent will be used."}),
                "latent": ("LATENT", {"tooltip": "Optional latent input. Will be decoded for processing."}),
                "mask": ("MASK", {"tooltip": "Optional: Face mask for processing. If connected, internal detection is skipped."}),
                "clip": ("CLIP", {"tooltip": "Optional: Required only if using face prompts."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "processed_face", "side_by_side_comparison", "final_mask")
    FUNCTION = "process_face_complete"
    CATEGORY = "Forbidden Vision"
   
    
    
    def __init__(self):
        ensure_model_directories()
        self.face_detector = ForbiddenVisionFaceDetector()
        self.mask_processor = ForbiddenVisionMaskProcessor()
        self.last_sampling_key = None
        self.last_sampling_result = None
        self.upscaler_model = None
        self.upscaler_model_name = None
    

    def load_upscaler_model(self, model_name):
        clean_model_name_val = clean_model_name(model_name)
        
        if clean_model_name_val in ["Fast 4x (Bicubic AA)", "Fast 4x (Lanczos)"]:
            self.upscaler_model = None
            self.upscaler_model_name = clean_model_name_val
            return True

        if self.upscaler_model is not None and self.upscaler_model_name == clean_model_name_val:
            return True

        try:
            model_path = folder_paths.get_full_path("upscale_models", clean_model_name_val)
            if model_path is None:
                print(f"Upscaler model '{clean_model_name_val}' not found.")
                return False
            
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            
            self.upscaler_model = upscaler_loader.load_model(clean_model_name_val)[0]
            self.upscaler_model_name = clean_model_name_val
            
            return True
            
        except model_management.InterruptProcessingException:
            raise
        except KeyError:
            print("Error: Could not find 'UpscaleModelLoader' in nodes.NODE_CLASS_MAPPINGS.")
            return False
        except Exception as e:
            print(f"An error occurred loading upscaler model {clean_model_name_val}: {e}")
            self.upscaler_model = None
            self.upscaler_model_name = None
            return False

    def run_upscaler(self, image_np_uint8):
        upscaler_model_name = getattr(self, 'upscaler_model_name', None)
        
        if upscaler_model_name == "Fast 4x (Bicubic AA)":
            return self.fast_upscale_bicubic(image_np_uint8)
        elif upscaler_model_name == "Fast 4x (Lanczos)":
            return self.fast_upscale_lanczos(image_np_uint8)
        elif self.upscaler_model is None:
            return image_np_uint8
        else:
            return self.run_ai_upscaler(image_np_uint8)

    def run_ai_upscaler(self, image_np_uint8):
        if self.upscaler_model is None:
            return image_np_uint8
        
        image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0).unsqueeze(0)
        
        try:
            ImageUpscalerClass = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']
            upscaler_node = ImageUpscalerClass()

            with torch.no_grad():
                upscaled_tensor = upscaler_node.upscale(
                    upscale_model=self.upscaler_model,
                    image=image_tensor
                )[0]

        except model_management.InterruptProcessingException:
            raise
        except KeyError:
            print("Error: Could not find 'ImageUpscaleWithModel' in nodes.NODE_CLASS_MAPPINGS.")
            return image_np_uint8
        except Exception as e:
            print(f"Error during upscaling process: {e}. Falling back to standard resize.")
            return image_np_uint8

        upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy()
        upscaled_np_uint8 = (np.clip(upscaled_np, 0, 1) * 255.0).round().astype(np.uint8)

        return upscaled_np_uint8
    def _perform_color_correction_gpu(self, processed_face_tensor, original_crop_tensor, correction_strength):
        try:
            check_for_interruption()
            
            processed_face_permuted = processed_face_tensor.permute(0, 3, 1, 2).float()
            original_crop_permuted = original_crop_tensor.permute(0, 3, 1, 2).float()

            original_face_resized = F.interpolate(
                original_crop_permuted, 
                size=processed_face_permuted.shape[2:], 
                mode='bicubic', 
                align_corners=False, 
                antialias=True
            )

            processed_lab = kornia.color.rgb_to_lab(processed_face_permuted)
            target_lab = kornia.color.rgb_to_lab(original_face_resized)

            source_mean = torch.mean(processed_lab, dim=(2, 3), keepdim=True)
            source_std = torch.std(processed_lab, dim=(2, 3), keepdim=True)
            target_mean = torch.mean(target_lab, dim=(2, 3), keepdim=True)
            target_std = torch.std(target_lab, dim=(2, 3), keepdim=True)
            
            result_lab = (processed_lab - source_mean) * (target_std / (source_std + 1e-6)) + target_mean
            
            corrected_face_permuted = kornia.color.lab_to_rgb(result_lab)

            if correction_strength < 1.0:
                corrected_face_permuted = processed_face_permuted * (1.0 - correction_strength) + corrected_face_permuted * correction_strength
            
            final_tensor = torch.clamp(corrected_face_permuted, 0.0, 1.0).permute(0, 2, 3, 1)

            return final_tensor

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in GPU color correction: {e}. Returning original processed tensor.")
            return processed_face_tensor

    def create_compositing_blend_mask_gpu(self, clean_sam_mask_tensor, blend_softness):
        try:
            check_for_interruption()
            
            if blend_softness <= 0:
                return (clean_sam_mask_tensor > 0.5).float()

            h, w = clean_sam_mask_tensor.shape[1], clean_sam_mask_tensor.shape[2]
            
            mask_4d = clean_sam_mask_tensor.unsqueeze(0)
            
            core_mask = (mask_4d > 0.5).float()
            
            mask_size = max(w, h, 1)
            blend_ratio = blend_softness / 400.0
            
            expand_amount = int(mask_size * blend_ratio * 0.8)
            if expand_amount > 0:
                expand_kernel_size = expand_amount * 2 + 1
                expand_kernel = torch.ones(expand_kernel_size, expand_kernel_size, device=mask_4d.device)
                expanded_for_blur = kornia.morphology.dilation(core_mask, expand_kernel)
            else:
                expanded_for_blur = core_mask

            blur_amount = int(mask_size * blend_ratio * 1.2)
            blur_kernel_size = max(3, blur_amount * 2 + 1)
            sigma = blur_kernel_size / 3.0
            feathering_zone = kornia.filters.gaussian_blur2d(
                expanded_for_blur, (blur_kernel_size, blur_kernel_size), (sigma, sigma)
            )

            final_mask = torch.max(core_mask, feathering_zone)

            fade_pixels = int(min(h, w) * 0.02)
            if fade_pixels > 0:
                fade_y = torch.linspace(0.0, 1.0, steps=fade_pixels, device=final_mask.device)
                fade_x = torch.linspace(0.0, 1.0, steps=fade_pixels, device=final_mask.device)
                final_mask[:, :, :fade_pixels, :] *= fade_y.view(1, 1, -1, 1)
                final_mask[:, :, h-fade_pixels:, :] *= torch.flip(fade_y, [0]).view(1, 1, -1, 1)
                final_mask[:, :, :, :fade_pixels] *= fade_x.view(1, 1, 1, -1)
                final_mask[:, :, :, w-fade_pixels:] *= torch.flip(fade_x, [0]).view(1, 1, 1, -1)

            final_mask[:, :, 0, :], final_mask[:, :, -1, :], final_mask[:, :, :, 0], final_mask[:, :, :, -1] = 0, 0, 0, 0

            return torch.clamp(final_mask.squeeze(0), 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating GPU hybrid blend mask: {e}")
            return (clean_sam_mask_tensor > 0.5).float()
    def combine_all_faces_to_final_image(self, original_image_tensor, all_processed_faces, all_restore_info, blend_softness, enable_color_correction=False, color_correction_strength=1.0):
        try:
            check_for_interruption()
            
            if not all_processed_faces or not all_restore_info:
                return original_image_tensor

            device = original_image_tensor.device
            h, w = original_image_tensor.shape[1], original_image_tensor.shape[2]

            final_canvas_tensor = original_image_tensor.clone()
            
            unified_compositing_mask = torch.zeros((1, h, w), device=device)

            for processed_face_tensor, restore_info in zip(all_processed_faces, all_restore_info):
                check_for_interruption()
                
                if isinstance(restore_info, list):
                    restore_info = restore_info[0]

                crop_coords = restore_info["crop_coords"]
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                actual_crop_w, actual_crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

                if actual_crop_w <= 0 or actual_crop_h <= 0:
                    continue

                processed_face_tensor = processed_face_tensor.to(device)

                if enable_color_correction and color_correction_strength > 0.0:
                    original_crop_tensor = original_image_tensor[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
                    processed_face_tensor = self._perform_color_correction_gpu(
                        processed_face_tensor, original_crop_tensor, color_correction_strength
                    )

                resized_processed_face = F.interpolate(
                    processed_face_tensor.permute(0, 3, 1, 2),
                    size=(actual_crop_h, actual_crop_w),
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)
                
                temp_face_canvas = torch.zeros_like(final_canvas_tensor)
                temp_face_canvas[:, crop_y1:crop_y2, crop_x1:crop_x2, :] = resized_processed_face

                full_blend_mask = torch.from_numpy(restore_info.get("blend_mask")).to(device)
                if full_blend_mask is None:
                    paste_mask_crop = torch.ones((actual_crop_h, actual_crop_w), device=device)
                else:
                    paste_mask_crop = full_blend_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                compositing_mask_crop = self.create_compositing_blend_mask_gpu(paste_mask_crop.unsqueeze(0), blend_softness)
                
                temp_mask_canvas = torch.zeros_like(unified_compositing_mask)
                temp_mask_canvas[:, crop_y1:crop_y2, crop_x1:crop_x2] = compositing_mask_crop
                
                final_canvas_tensor = temp_face_canvas * temp_mask_canvas.unsqueeze(-1) + final_canvas_tensor * (1.0 - temp_mask_canvas.unsqueeze(-1))

            check_for_interruption()
            
            return final_canvas_tensor

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error combining faces to final image on GPU: {e}")
            return original_image_tensor

    def process_face_complete(self, model, vae, positive, negative, 
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                        face_selection, processing_resolution, use_adaptive_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                        face_positive_prompt, replace_positive_prompt, face_negative_prompt, replace_negative_prompt, exclusions,
                        blend_softness, mask_expansion,
                        sampling_mask_blur_size, sampling_mask_blur_strength,
                        enable_vertical_flip, enable_color_correction,
                        bbox_model, bbox_model_B, yolo_seg_model, yolo_seg_model_B,
                        image=None, mask=None, clip=None, latent=None):
        try:
            check_for_interruption()

            input_image = None
            if latent is not None and "samples" in latent:
                print("[Face Processor] Latent input detected. Decoding for processing.")
                with torch.no_grad():
                    input_image = vae.decode(latent["samples"])
                    if input_image.min() < -0.5:
                        input_image = (input_image + 1.0) / 2.0
                    input_image = torch.clamp(input_image, 0.0, 1.0)
            else:
                input_image = image

            if input_image is None or model is None or vae is None:
                print("ERROR: Required inputs (image/latent, model, vae) are missing.")
                return self.create_safe_fallback_outputs(input_image, processing_resolution)

            original_image = input_image
            final_positive_for_face, final_negative_for_face = positive, negative

            if clip and (exclusions or face_positive_prompt or face_negative_prompt):
                
                def extract_original_text(conditioning):
                    try:
                        if conditioning and len(conditioning) > 0 and len(conditioning[0]) > 1:
                            metadata = conditioning[0][1].get("forbidden_vision_metadata")
                            if metadata and isinstance(metadata, dict):
                                return metadata.get('original_text', '')
                    except Exception:
                        pass
                    return ""

                processor = ExclusionProcessor()
                
                pos_base_text = extract_original_text(positive)
                if pos_base_text or face_positive_prompt:
                    processed_text = processor.process(pos_base_text, exclusions).strip()
                    face_prompt = face_positive_prompt.strip()
                    
                    if replace_positive_prompt and face_prompt:
                        final_pos_text = face_prompt
                    else:
                        final_pos_text = ", ".join(filter(None, [face_prompt, processed_text]))

                    pos_tokens = clip.tokenize(final_pos_text)
                    cond, pooled = clip.encode_from_tokens(pos_tokens, return_pooled=True)
                    final_positive_for_face = [[cond, {"pooled_output": pooled}]]

                neg_base_text = extract_original_text(negative)
                if neg_base_text or face_negative_prompt:
                    processed_text = processor.process(neg_base_text, exclusions).strip()
                    face_prompt = face_negative_prompt.strip()

                    if replace_negative_prompt and face_prompt:
                        final_neg_text = face_prompt
                    else:
                        final_neg_text = ", ".join(filter(None, [face_prompt, processed_text]))

                    neg_tokens = clip.tokenize(final_neg_text)
                    cond, pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
                    final_negative_for_face = [[cond, {"pooled_output": pooled}]]
            
            elif not clip and (exclusions or face_positive_prompt or face_negative_prompt):
                print("[Face Processor] Warning: Face prompts or exclusions were provided, but no CLIP model was connected. These options will be ignored.")


            face_masks = []
            if mask is not None:
                face_masks = [mask]
            else:
                np_masks = self.face_detector.detect_faces(
                    image_tensor=input_image, 
                    bbox_model_name=bbox_model,
                    bbox_model_B_name=bbox_model_B,
                    sam_model_name=yolo_seg_model,
                    sam_model_B_name=yolo_seg_model_B,
                    detection_confidence=0.5, 
                    sam_threshold=0.4, 
                    face_selection=face_selection
                )
                if np_masks:
                    face_masks = [torch.from_numpy(m).unsqueeze(0) for m in np_masks]

            if not face_masks:
                print("ERROR: No face masks found. Cannot proceed.")
                return self.create_safe_fallback_outputs(input_image, processing_resolution)
            
            self.last_sampling_key = None 
            
            processing_image = torch.flip(input_image, dims=[1]) if enable_vertical_flip else input_image.clone()
            if enable_vertical_flip and mask is not None:
                face_masks = [torch.flip(m, dims=[1]) for m in face_masks]
            
            all_processed_faces, all_restore_info = [], []
            
            for i, face_mask in enumerate(face_masks):
                check_for_interruption()
                processed_result = self.process_single_face_unified(
                    processing_image, face_mask, model, vae, final_positive_for_face, final_negative_for_face,
                    steps, cfg_scale, sampler, scheduler, denoise_strength, seed + i,
                    processing_resolution, use_adaptive_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                    mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength
                )
                if processed_result:
                    _, processed_face, restore_info = processed_result
                    all_processed_faces.append(processed_face)
                    all_restore_info.append(restore_info)
                    if len(face_masks) > 1 and i < len(face_masks) - 1 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not all_processed_faces:
                print("ERROR: All face processing failed. Returning original image.")
                return self.create_safe_fallback_outputs(original_image, processing_resolution)

            compositing_base_image = torch.flip(input_image, dims=[1]) if enable_vertical_flip else original_image.clone()
            final_image = self.combine_all_faces_to_final_image(
                compositing_base_image, all_processed_faces, all_restore_info, 
                blend_softness, enable_color_correction, 1.0
            )
            
            processed_face_output = self.create_combined_face_output(all_processed_faces, processing_resolution)
            side_by_side = self.create_unified_comparison(original_image, all_processed_faces, all_restore_info, processing_resolution)
            final_mask = self.create_unified_mask(all_restore_info, original_image)
            
            if enable_vertical_flip:
                final_image = torch.flip(final_image, dims=[1])
                processed_face_output = torch.flip(processed_face_output, dims=[1])
                side_by_side = torch.flip(side_by_side, dims=[1])
                final_mask = torch.flip(final_mask, dims=[1])

            return (final_image, processed_face_output, side_by_side, final_mask)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"An error occurred during the main face processing workflow: {e}")
            (fallback_image, fallback_processed, fallback_comparison, fallback_mask) = self.create_safe_fallback_outputs(input_image, processing_resolution)
            if enable_vertical_flip and fallback_image is not None:
                fallback_image = torch.flip(fallback_image, dims=[1])
                fallback_processed = torch.flip(fallback_processed, dims=[1])
                fallback_comparison = torch.flip(fallback_comparison, dims=[1])
                fallback_mask = torch.flip(fallback_mask, dims=[1])
            return (fallback_image, fallback_processed, fallback_comparison, fallback_mask)
    
    def _get_adaptive_resolution(self, mask_tensor):
        try:
            non_zero_coords = torch.where(mask_tensor > 0.5)
            if len(non_zero_coords[0]) == 0 or len(non_zero_coords[1]) == 0:
                return (1024, 1024) 

            y_min, y_max = non_zero_coords[1].min(), non_zero_coords[1].max()
            x_min, x_max = non_zero_coords[2].min(), non_zero_coords[2].max()

            mask_h = y_max - y_min
            mask_w = x_max - x_min
            
            if mask_h == 0 or mask_w == 0:
                return (1024, 1024)

            aspect_ratio = float(mask_w) / float(mask_h)
            
            best_match = (1024, 1024)
            min_ratio_diff = float('inf')

            for res_w, res_h in self.SDXL_RESOLUTIONS.values():
                bucket_ratio = float(res_w) / float(res_h)
                diff = abs(aspect_ratio - bucket_ratio)
                
                if diff < min_ratio_diff:
                    min_ratio_diff = diff
                    best_match = (res_w, res_h)
            
            return best_match
        except Exception as e:
            print(f"Warning: Could not determine adaptive resolution due to error: {e}. Falling back to 1024x1024.")
            return (1024, 1024)
        
    def _hash_conditioning(self, conditioning):
        try:
            if not conditioning:
                return 0
            
            final_hash = 0
            for cond_item in conditioning:
                tensor_hash = 0
                if len(cond_item) > 0 and torch.is_tensor(cond_item[0]):
                    tensor = cond_item[0]
                    sample = tensor.flatten()[::max(1, tensor.numel() // 50)][:100]
                    tensor_hash = hash(sample.cpu().numpy().round(4).tobytes())

                dict_hash = 0
                if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                    for key, value in sorted(cond_item[1].items()):
                        key_hash = hash(key)
                        value_hash = 0
                        if torch.is_tensor(value):
                            sample = value.flatten()[::max(1, value.numel() // 20)][:50]
                            value_hash = hash(sample.cpu().numpy().round(4).tobytes())
                        else:
                            try:
                                value_hash = hash(value)
                            except TypeError:
                                value_hash = hash(str(value))
                        
                        dict_hash = (dict_hash * 31) + hash((key_hash, value_hash))

                item_hash = hash((tensor_hash, dict_hash))
                final_hash = (final_hash * 31) + item_hash
                
            return final_hash
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Warning: Could not fully hash conditioning, caching might be less reliable. Error: {e}")
            return random.randint(0, 0xffffffffffffffff)

    def get_sampling_cache_key_only(self, image, mask, model, vae, final_positive, final_negative,
                                steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                                processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                                mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength,
                                enable_vertical_flip, face_selection,
                                bbox_model, yolo_seg_model, yolo_seg_model_B):
        try:
            image_sample = image[0, ::96, ::96, :].flatten()[:50] if image.numel() > 2500 else image.flatten()[:50]
            image_hash = hash(tuple(image_sample.cpu().numpy().round(3)))
            
            mask_hash = 0
            if mask is not None:
                mask_sample = mask[0, ::96, ::96].flatten()[:50] if mask.numel() > 2500 else mask.flatten()[:50]
                mask_hash = hash(tuple(mask_sample.cpu().numpy().round(3)))
            else:
                mask_hash = hash((0.5, 0.4, bbox_model, yolo_seg_model, yolo_seg_model_B))

            model_hash = id(model)
            vae_hash = id(vae)
            
            pos_hash = self._hash_conditioning(final_positive)
            neg_hash = self._hash_conditioning(final_negative)
            
            return (
                image_hash, mask_hash, tuple(image.shape),
                model_hash, vae_hash, pos_hash, neg_hash,
                steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength,
                enable_vertical_flip, face_selection
            )
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error generating sampling cache key: {e}")
            return None
    
    def process_single_face_unified(self, image, mask, model, vae, final_positive, final_negative,
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                        processing_resolution, use_adaptive_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                        mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength):
        try:
            if mask is None:
                return None

            if use_adaptive_resolution:
                target_resolution = self._get_adaptive_resolution(mask)
                print(f"[Face Processor] Adaptive resolution: Mask aspect ratio matched to {target_resolution[0]}x{target_resolution[1]}.")
            else:
                target_resolution = (processing_resolution, processing_resolution)

            cropped_face, sampler_mask_batch, restore_info = self.mask_processor.process_and_crop(
                image_tensor=image, 
                mask_tensor=mask, 
                crop_padding=crop_padding, 
                processing_resolution=target_resolution, 
                mask_expansion=mask_expansion,
                enable_pre_upscale=enable_pre_upscale,
                upscaler_model_name=upscaler_model,
                upscaler_loader_callback=self.load_upscaler_model,
                upscaler_run_callback=self.run_upscaler
            )
            
            if self.is_empty_detection(cropped_face, restore_info):
                return None
            
            processed_latent = self.run_inpaint_sampling(
                cropped_face, sampler_mask_batch, model, vae, final_positive, final_negative,
                steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                sampling_mask_blur_size, sampling_mask_blur_strength
            )
            
            with torch.no_grad():
                processed_face_batch = vae.decode(processed_latent["samples"])
                if processed_face_batch.min() < -0.5: 
                    processed_face_batch = (processed_face_batch + 1.0) / 2.0
                processed_face_batch = torch.clamp(processed_face_batch, 0.0, 1.0)
            
            
            return (image, processed_face_batch, restore_info)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error processing single face: {e}")
            return None

    def create_combined_face_output(self, processed_faces, processing_resolution):
        try:
            check_for_interruption()

            if not processed_faces:
                return torch.zeros((1, processing_resolution, processing_resolution, 3), device=model_management.get_torch_device())
            
            if len(processed_faces) == 1:
                return processed_faces[0]
            
            device = processed_faces[0].device
            face_tensors = []
            for face_tensor in processed_faces:
                face_permuted = face_tensor.to(device).permute(0, 3, 1, 2)
                resized_face = F.interpolate(
                    face_permuted, 
                    size=(processing_resolution, processing_resolution), 
                    mode='bicubic', 
                    align_corners=False,
                    antialias=True
                )
                face_tensors.append(resized_face.permute(0, 2, 3, 1))
            
            if len(face_tensors) == 2:
                combined = torch.cat(face_tensors, dim=2)
            else:
                rows = []
                for i in range(0, len(face_tensors), 2):
                    row_faces = face_tensors[i:i+2]
                    if len(row_faces) == 1:
                        black_tensor = torch.zeros_like(row_faces[0])
                        row_faces.append(black_tensor)
                    row = torch.cat(row_faces, dim=2)
                    rows.append(row)
                combined = torch.cat(rows, dim=1)
            
            return combined
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating GPU-based combined face output: {e}")
            return processed_faces[0] if processed_faces else torch.zeros((1, processing_resolution, processing_resolution, 3), device=model_management.get_torch_device())

    def create_unified_comparison(self, original_image, processed_faces, restore_info_list, processing_resolution):
        try:
            check_for_interruption()

            if not processed_faces or not restore_info_list:
                return original_image
            
            device = original_image.device
            comparisons = []
            
            for processed_face, restore_info in zip(processed_faces, restore_info_list):
                crop_coords = restore_info["crop_coords"]
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue
                
                original_crop = original_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
                
                original_resized = F.interpolate(
                    original_crop.permute(0, 3, 1, 2),
                    size=(processing_resolution, processing_resolution),
                    mode='bicubic', align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)

                processed_resized = F.interpolate(
                    processed_face.to(device).permute(0, 3, 1, 2),
                    size=(processing_resolution, processing_resolution),
                    mode='bicubic', align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)
                
                face_comparison = torch.cat([original_resized, processed_resized], dim=2)
                comparisons.append(face_comparison)
            
            if not comparisons:
                return original_image
            
            final_comparison = torch.cat(comparisons, dim=1)
            
            return final_comparison
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating GPU-based unified comparison: {e}")
            return original_image

    def create_unified_mask(self, restore_info_list, image):
        try:
            check_for_interruption()

            image_shape = image.shape
            original_height, original_width = image_shape[1], image_shape[2]
            device = image.device
            
            if not restore_info_list:
                return torch.zeros((1, original_height, original_width), dtype=torch.float32, device=device)
            
            combined_mask = torch.zeros((1, original_height, original_width), dtype=torch.float32, device=device)
            
            for restore_info in restore_info_list:
                blend_mask_np = restore_info.get("blend_mask")
                if blend_mask_np is not None:
                    blend_mask_tensor = torch.from_numpy(blend_mask_np).to(device)
                    combined_mask = torch.maximum(combined_mask, blend_mask_tensor)

            combined_mask_np = combined_mask.squeeze(0).cpu().numpy()
            
            polished_mask_np = self.mask_processor.polish_mask(combined_mask_np)
            
            return torch.from_numpy(polished_mask_np).unsqueeze(0).to(device)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating unified mask: {e}")
            image_shape = image.shape
            return torch.zeros((1, image_shape[1], image_shape[2]), dtype=torch.float32, device=image.device)
    
    def process_single_face_sampling(self, cropped_face, face_mask, model, vae, positive, negative,
                                    steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                                    sampling_mask_blur_size, sampling_mask_blur_strength):
        try:
            face_latent = self.encode_image_to_latent(cropped_face, vae)
            latent_samples = face_latent["samples"]
            
            B, C, H, W = latent_samples.shape
            resized_mask = self.process_inpaint_mask(face_mask, H, W, latent_samples.device, 
                                        sampling_mask_blur_size, sampling_mask_blur_strength)

            conditioned_positive, conditioned_negative = self.prepare_inpaint_conditioning(
                positive, negative, latent_samples, resized_mask.unsqueeze(1)
            )
            
            sampled_latent = self.run_ksampler(
                model, conditioned_positive, conditioned_negative, face_latent,
                steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                denoise_mask=resized_mask
            )
            
            return sampled_latent
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in single face sampling: {e}")
            raise e
    
    def create_safe_fallback_outputs(self, image, processing_resolution):
        try:
            device = model_management.get_torch_device()
            
            final_output = image.clone().to(device) if image is not None else torch.zeros((1, 512, 512, 3), dtype=torch.float32, device=device)
            
            empty_processed = torch.zeros((1, processing_resolution, processing_resolution, 3), dtype=torch.float32, device=device)

            if image is not None:
                h, w = image.shape[1], image.shape[2]
                black_side = torch.zeros_like(image)
                empty_comparison = torch.cat([image, black_side], dim=2)
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
            else:
                h, w = 512, 512
                empty_comparison = torch.zeros((1, h, w * 2, 3), dtype=torch.float32, device=device)
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
            
            return (final_output, empty_processed, empty_comparison, empty_mask)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating safe fallback outputs: {e}")
            device = model_management.get_torch_device()
            fallback_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device=device)
            fallback_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=device)
            fallback_comp = torch.zeros((1, 512, 1024, 3), dtype=torch.float32, device=device)
            return (fallback_img, fallback_img, fallback_comp, fallback_mask)
    

    
    def is_empty_detection(self, cropped_face, restore_info):
        try:
            if isinstance(restore_info, list):
                return len(restore_info) == 0 or all(info.get("original_image_size", (0, 0))[0] == 0 for info in restore_info)
            else:
                return restore_info.get("original_image_size", (0, 0))[0] == 0
        except model_management.InterruptProcessingException:
            raise
        except:
            return True
    
    def run_inpaint_sampling(self, cropped_face, face_mask, model, vae, positive, negative,
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                        sampling_mask_blur_size, sampling_mask_blur_strength):
        try:
            result = self.process_single_face_sampling(
                cropped_face, face_mask, model, vae, positive, negative,
                steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                sampling_mask_blur_size, sampling_mask_blur_strength
            )
            return result
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in inpaint sampling: {e}")
            return {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)}
  
    def process_inpaint_mask(self, face_mask, latent_height, latent_width, device, blur_size, blur_strength):
        try:
            check_for_interruption()
            
            mask_4d = face_mask.reshape(1, 1, face_mask.shape[-2], face_mask.shape[-1])
            resized_mask_4d = F.interpolate(mask_4d, size=(latent_height, latent_width), mode='bilinear', align_corners=False)

            if blur_size > 1 and blur_strength > 0:
                import math
                
                if blur_size % 2 == 0:
                    blur_size += 1
                
                base_sigma = (blur_size - 1) / 8.0
                actual_sigma = base_sigma * (1 + math.tanh(blur_strength - 1) * 2)
                
                blurred_mask_4d = kornia.filters.gaussian_blur2d(
                    resized_mask_4d, (blur_size, blur_size), (actual_sigma, actual_sigma)
                )
                
                return blurred_mask_4d.squeeze(1)

            return resized_mask_4d.squeeze(1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error processing inpaint mask on GPU: {e}")
            if 'resized_mask_4d' in locals():
                return resized_mask_4d.squeeze(1)
            else:
                return torch.zeros((face_mask.shape[0], latent_height, latent_width), device=device)

    def encode_image_to_latent(self, image, vae):
        try:
            with torch.no_grad():
                encoded = vae.encode(image)
            
            return {"samples": encoded}
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error encoding image to latent: {e}")
            raise e

    def prepare_inpaint_conditioning(self, positive, negative, latent_samples, resized_mask):
        try:
            conditioned_positive = []
            for p in positive:
                new_cond_dict = p[1].copy()
                new_cond_dict["concat_latent_image"] = latent_samples
                new_cond_dict["concat_mask"] = resized_mask
                conditioned_positive.append([p[0], new_cond_dict])
            
            conditioned_negative = []
            for n in negative:
                new_cond_dict = n[1].copy()
                new_cond_dict["concat_latent_image"] = latent_samples
                new_cond_dict["concat_mask"] = resized_mask
                conditioned_negative.append([n[0], new_cond_dict])
                
            return conditioned_positive, conditioned_negative
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error preparing inpaint conditioning: {e}")
            return positive, negative

    def run_ksampler(self, model, positive, negative, latent, steps, cfg, sampler_name, scheduler, denoise, seed, denoise_mask=None):
        try:
            check_for_interruption()
            device = model_management.get_torch_device()
            latent_image = latent["samples"]

            if denoise_mask is not None:
                denoise_mask = denoise_mask.to(device)

            try:
                noise = comfy.sample.prepare_noise(latent_image, seed, device=device)
            except TypeError:
                noise = comfy.sample.prepare_noise(latent_image, seed)
                noise = noise.to(device)
            positive_cond = self.prepare_conditioning_for_sampling(positive, device)
            negative_cond = self.prepare_conditioning_for_sampling(negative, device)

            try:
                previewer = latent_preview.get_previewer(device, model.model.latent_format)
            except:
                previewer = None

            pbar = comfy.utils.ProgressBar(steps)

            def preview_callback(step, x0, x, total_steps):
                preview_bytes = None
                if previewer:
                    try:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    except:
                        pass
                pbar.update_absolute(step + 1, total_steps, preview_bytes)

            
            sampler = comfy.samplers.KSampler(
                model, steps=steps, device=device, sampler=sampler_name,
                scheduler=scheduler, denoise=denoise, model_options=model.model_options,
            )

            samples = sampler.sample(
                noise, positive_cond, negative_cond, cfg=cfg,
                latent_image=latent_image, denoise_mask=denoise_mask,
                start_step=0, last_step=steps, force_full_denoise=False,
                callback=preview_callback,
            )
            
            return {"samples": samples}

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in KSampler: {e}")
            latent_shape = latent.get("samples", torch.zeros((1, 4, 64, 64))).shape
            return {"samples": torch.zeros(latent_shape, device=model_management.get_torch_device())}

    def prepare_conditioning_for_sampling(self, conditioning, device):
        try:
            if not conditioning:
                return conditioning
                
            prepared_conditioning = []
            
            for cond_item in conditioning:
                if cond_item[0].device == device:
                    cond_tensor = cond_item[0]
                else:
                    cond_tensor = cond_item[0].to(device)
                
                cond_dict = {}
                for key, value in cond_item[1].items():
                    if torch.is_tensor(value) and value.device != device:
                        cond_dict[key] = value.to(device)
                    else:
                        cond_dict[key] = value
                
                prepared_conditioning.append([cond_tensor, cond_dict])
            
            return prepared_conditioning
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error preparing conditioning: {e}")
            return conditioning
    
    def fast_upscale_bicubic(self, image_np_uint8):
        try:
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0).unsqueeze(0)
            
            with torch.no_grad():
                upscaled_tensor = F.interpolate(
                    image_tensor.permute(0, 3, 1, 2),
                    scale_factor=4,
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)
            
            upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy()
            upscaled_np_uint8 = (np.clip(upscaled_np, 0, 1) * 255.0).round().astype(np.uint8)
            
            cleaned_np_uint8 = self.clean_interpolation_edges(upscaled_np_uint8)
            
            return cleaned_np_uint8
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in fast bicubic upscaling: {e}. Returning original image.")
            return image_np_uint8

    def fast_upscale_lanczos(self, image_np_uint8):
        try:
            h, w = image_np_uint8.shape[:2]
            new_h, new_w = h * 4, w * 4
            
            upscaled_np_uint8 = cv2.resize(
                image_np_uint8, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            cleaned_np_uint8 = self.clean_interpolation_edges(upscaled_np_uint8)
            
            return cleaned_np_uint8
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in fast Lanczos upscaling: {e}. Returning original image.")
            return image_np_uint8
    
    def clean_interpolation_edges(self, image_np_uint8):
        try:
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            edge_size = max(8, min(h, w) // 32)
            
            mask = torch.ones_like(image_tensor[:, :1, :, :])
            
            mask[:, :, :edge_size, :] = torch.linspace(0, 1, edge_size).view(1, 1, edge_size, 1)
            mask[:, :, h-edge_size:, :] = torch.linspace(1, 0, edge_size).view(1, 1, edge_size, 1)
            mask[:, :, :, :edge_size] = torch.minimum(mask[:, :, :, :edge_size], 
                                                    torch.linspace(0, 1, edge_size).view(1, 1, 1, edge_size))
            mask[:, :, :, w-edge_size:] = torch.minimum(mask[:, :, :, w-edge_size:], 
                                                    torch.linspace(1, 0, edge_size).view(1, 1, 1, edge_size))
            
            kernel_size = max(3, edge_size // 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = kernel_size / 3.0
            
            blurred = kornia.filters.gaussian_blur2d(image_tensor, (kernel_size, kernel_size), (sigma, sigma))
            
            cleaned = image_tensor * mask + blurred * (1 - mask)
            
            cleaned_np = cleaned.squeeze(0).permute(1, 2, 0).cpu().numpy()
            cleaned_np_uint8 = (np.clip(cleaned_np, 0, 1) * 255.0).round().astype(np.uint8)
            
            return cleaned_np_uint8
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error cleaning interpolation edges: {e}. Returning original image.")
            return image_np_uint8