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
from .utils import check_for_interruption, get_ordered_upscaler_model_list, ensure_model_directories, clean_model_name

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

    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()
        
        default_upscaler = "Fast 4x (Lanczos)"
        if "Fast 4x (Lanczos)" not in upscaler_models and upscaler_models:
            default_upscaler = upscaler_models[0]
        
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "denoise_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                "face_selection": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "0=All faces, 1=1st face, etc."}),
                "detection_confidence": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Face detection confidence threshold."}),
                "manual_rotation": (["None", "90° CW", "90° CCW", "180°"], {"default": "None", "tooltip": "Manually rotate face crop before processing"}),
                "processing_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "The resolution for processing."}),

                "enable_pre_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Enable to upscale small faces with an AI model before processing."}),
                "upscaler_model": (upscaler_models, {"default": default_upscaler, "tooltip": "The model used for pre-upscaling small faces."}),
                "crop_padding": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Padding added to the face region before inpaint."}),
                
                "face_positive_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Additional positive prompt for face generation"}),
                "replace_positive_prompt": ("BOOLEAN", {"default": False}),
                "face_negative_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Additional negative prompt for face generation"}),
                "replace_negative_prompt": ("BOOLEAN", {"default": False}),   
                "exclusions": ("STRING", {"multiline": True, "default": "", "placeholder": "e.g. glasses, scar, wrinkles\nTags to remove from main prompt for face generation...", "tooltip": "Words/tags to remove from the main prompt specifically for the face processing step."}),

                "blend_softness": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "mask_expansion": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "sampling_mask_blur_size": ("INT", {"default": 21, "min": 1, "max": 101, "step": 2}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.1}),
                
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_segmentation": ("BOOLEAN", {"default": True, "tooltip": "Use AI segmentation. If disabled, creates oval masks."}),
                "enable_differential_diffusion": ("BOOLEAN", {"default": True, "tooltip": "Better blending. At high noise, the mask allows structure changes; at low noise, it locks the background."}),
                "enable_lightness_rescue": ("BOOLEAN", {"default": True, "tooltip": "If the generated face is darker than original, brighten it and run a micro-pass to fix exposure."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image input. If latent is also provided, latent will be used."}),
                "latent": ("LATENT", {"tooltip": "Optional latent input. Will be decoded for processing."}),
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

        self.upscaler_model = None
        self.upscaler_model_name = None
    
    def differential_diffusion_function(self, sigma, denoise_mask, extra_options):
        try:
            model = extra_options["model"]
            step_sigmas = extra_options["sigmas"]

            inner_model = getattr(model, "inner_model", model)

            sampling = inner_model.model_sampling

            sigma_to = sampling.sigma_min
            if step_sigmas[-1] > sigma_to:
                sigma_to = step_sigmas[-1]
            sigma_from = step_sigmas[0]

            ts_from = sampling.timestep(sigma_from)
            ts_to = sampling.timestep(sigma_to)
            current_ts = sampling.timestep(sigma[0])

            threshold = (current_ts - ts_to) / (ts_from - ts_to)

            binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

            strength = 1.0

            if strength and strength < 1.0:
                blended_mask = strength * binary_mask + (1.0 - strength) * denoise_mask
                return blended_mask
            else:
                return binary_mask

        except Exception as e:
            print(f"[ForbiddenVision] Differential diffusion mask error: {e}")
            return denoise_mask

    def load_upscaler_model(self, model_name):
        clean_model_name_val = clean_model_name(model_name)
        
        if clean_model_name_val in ["Fast 4x (Bicubic AA)", "Fast 4x (Lanczos)", "Fast 2x (Bicubic AA)", "Fast 2x (Lanczos)"]:
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
            return self.fast_upscale_bicubic(image_np_uint8, scale=4)
        elif upscaler_model_name == "Fast 4x (Lanczos)":
            return self.fast_upscale_lanczos(image_np_uint8, scale=4)
        elif upscaler_model_name == "Fast 2x (Bicubic AA)": 
            return self.fast_upscale_bicubic(image_np_uint8, scale=2)
        elif upscaler_model_name == "Fast 2x (Lanczos)":
            return self.fast_upscale_lanczos(image_np_uint8, scale=2)
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
            
            upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy()
            upscaled_np_uint8 = (np.clip(upscaled_np, 0, 1) * 255.0).astype(np.uint8)
            return upscaled_np_uint8

        except model_management.InterruptProcessingException:
            raise
        except KeyError:
            print("Error: Could not find 'ImageUpscaleWithModel' in nodes.NODE_CLASS_MAPPINGS.")
            return image_np_uint8
        except Exception as e:
            print(f"Error during upscaling process: {e}. Falling back to standard resize.")
            return image_np_uint8

    def _perform_color_correction_gpu(self, processed_face_tensor, original_crop_tensor, correction_strength, rescue_mask=None):
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
            
            if rescue_mask is not None:
                if rescue_mask.dim() == 3:
                    rescue_mask = rescue_mask.unsqueeze(0)
                
                spatial_weight = rescue_mask.permute(0, 3, 1, 2)
                
                if spatial_weight.shape[2:] != processed_lab.shape[2:]:
                    spatial_weight = F.interpolate(spatial_weight, size=processed_lab.shape[2:], mode='bilinear')

                hybrid_target_mean = target_mean.expand_as(processed_lab).clone()
                
                target_L = target_mean[:, 0:1, :, :]
                source_L = source_mean[:, 0:1, :, :]
                
                mixed_L = source_L * spatial_weight + target_L * (1.0 - spatial_weight)
                
                hybrid_target_mean[:, 0:1, :, :] = mixed_L
                
                result_lab = (processed_lab - source_mean) * (target_std / (source_std + 1e-6)) + hybrid_target_mean
                
            else:
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
                    
                    rescue_mask = restore_info.get("rescue_mask", None)
                    if rescue_mask is not None:
                        rescue_mask = rescue_mask.to(device)

                    processed_face_tensor = self._perform_color_correction_gpu(
                        processed_face_tensor, 
                        original_crop_tensor, 
                        color_correction_strength,
                        rescue_mask=rescue_mask
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
    
    def check_and_rescue_brightness(self, original_crop, processed_crop, mask, restore_info, model, vae, positive, negative, scheduler, sampler, seed, sampling_mask_blur_size, sampling_mask_blur_strength):
        try:
            check_for_interruption()
            device = original_crop.device
            
            if original_crop.ndim == 4: original_crop = original_crop.squeeze(0)
            if processed_crop.ndim == 4: processed_crop = processed_crop.squeeze(0)
            if mask.ndim == 4: mask = mask.squeeze(0).squeeze(0)
            if mask.ndim == 3: mask = mask.squeeze(0)
            
            h, w = processed_crop.shape[0], processed_crop.shape[1]

            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 1, 3)
            
            center_h, center_w = int(h * 0.35), int(w * 0.35)
            h_start, w_start = (h - center_h) // 2, (w - center_w) // 2
            
            orig_center = original_crop[h_start:h_start+center_h, w_start:w_start+center_w, :]
            proc_center = processed_crop[h_start:h_start+center_h, w_start:w_start+center_w, :]
            
            orig_luma_center = (orig_center * weights).sum() / (center_h * center_w)
            proc_luma_center = (proc_center * weights).sum() / (center_h * center_w)

            threshold_ratio = 0.95 
            
            if proc_luma_center < orig_luma_center * threshold_ratio:
                mean_orig_val = orig_luma_center.item()
                mean_proc_val = proc_luma_center.item()
                
                print(f"[Face Processor] Lightness Rescue Triggered: Skin is too dark (Orig: {mean_orig_val:.3f}, Proc: {mean_proc_val:.3f}). Rescuing...")
                
                
                target_map_size = 128
                mask_4d = mask.unsqueeze(0).unsqueeze(0)
                small_mask = F.interpolate(mask_4d, size=(target_map_size, target_map_size), mode='bilinear', align_corners=False)
                
                kernel_size = int(target_map_size * 0.05) 
                if kernel_size % 2 == 0: kernel_size += 1
                if kernel_size < 3: kernel_size = 3
                eroded_small = -F.max_pool2d(-small_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
                
                blur_k = kernel_size * 2 + 1
                sigma = blur_k / 3.0
                soft_small = kornia.filters.gaussian_blur2d(eroded_small, (blur_k, blur_k), (sigma, sigma))
                
                soft_rescue_mask_4d = F.interpolate(soft_small, size=(h, w), mode='bilinear', align_corners=False)
                spatial_mask = torch.clamp(soft_rescue_mask_4d.squeeze(0).permute(1, 2, 0), 0.0, 1.0)
                
                
                proc_tensor_4d = processed_crop.unsqueeze(0).permute(0, 3, 1, 2)
                proc_lab = kornia.color.rgb_to_lab(proc_tensor_4d)
                
                center_patch_lab = proc_lab[:, 1:, h_start:h_start+center_h, w_start:w_start+center_w]
                mean_center_ab = center_patch_lab.mean(dim=(2, 3), keepdim=True)
                
                diff = proc_lab[:, 1:, :, :] - mean_center_ab
                dist = torch.norm(diff, dim=1, keepdim=True)
                
                sensitivity = 0.05 
                chroma_mask = torch.exp(-dist * sensitivity)
                chroma_mask = chroma_mask.squeeze(0).permute(1, 2, 0)
                
                final_rescue_mask = spatial_mask * chroma_mask
                
                restore_info['rescue_mask'] = final_rescue_mask
                
                safe_orig = max(mean_orig_val, 0.01)
                safe_proc = max(mean_proc_val, 0.01)
                gamma = float(np.log(safe_orig) / np.log(safe_proc))
                gamma = max(0.4, min(gamma, 1.0)) 
                
                
                current_luma = (processed_crop * weights).sum(dim=-1, keepdim=True)
                target_luma = torch.pow(current_luma, gamma)
                
                gain_map = target_luma / (current_luma + 1e-6)
                
                gamma_corrected_full = processed_crop * gain_map
                
                shadow_threshold_low = 0.05
                shadow_threshold_high = 0.20
                shadow_anchor = torch.clamp((current_luma - shadow_threshold_low) / (shadow_threshold_high - shadow_threshold_low), 0.0, 1.0)
                
                
                combined_weight = final_rescue_mask * shadow_anchor
                
                smart_gamma_face = processed_crop + (gamma_corrected_full - processed_crop) * combined_weight
                corrected_face = torch.clamp(smart_gamma_face, 0.0, 1.0)
                
                corrected_batch = corrected_face.unsqueeze(0)
                
                rescue_latent = self.run_inpaint_sampling(
                    corrected_batch, mask.unsqueeze(0).unsqueeze(0), 
                    model, vae, positive, negative,
                    steps=2, cfg_scale=1.0, 
                    sampler=sampler, scheduler=scheduler,
                    denoise_strength=0.05, seed=seed + 1,
                    sampling_mask_blur_size=sampling_mask_blur_size,
                    sampling_mask_blur_strength=sampling_mask_blur_strength
                )
                
                with torch.no_grad():
                    rescued_face = vae.decode(rescue_latent["samples"])
                    if rescued_face.min() < -0.5:
                        rescued_face = (rescued_face + 1.0) / 2.0
                    rescued_face = torch.clamp(rescued_face, 0.0, 1.0)
                
                return rescued_face
            
            return processed_crop.unsqueeze(0)
            
        except Exception as e:
            print(f"[Face Processor] Error in lightness rescue: {e}. Keeping original processed face.")
            if processed_crop.ndim == 3: return processed_crop.unsqueeze(0)
            return processed_crop

    def process_face_complete(self, model, vae, positive, negative, 
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                        face_selection, detection_confidence, manual_rotation, processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                        face_positive_prompt, replace_positive_prompt, face_negative_prompt, replace_negative_prompt, exclusions,
                        blend_softness, mask_expansion,
                        sampling_mask_blur_size, sampling_mask_blur_strength,
                        enable_color_correction, enable_segmentation, enable_differential_diffusion, enable_lightness_rescue,
                        image=None, clip=None, latent=None):
        try:
            check_for_interruption()

            input_image = None
            if latent is not None and "samples" in latent:
         
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
            processing_image = input_image.clone()
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
                    
                    new_cond_dict = positive[0][1].copy() if positive and len(positive) > 0 else {}
                    new_cond_dict["pooled_output"] = pooled
                    final_positive_for_face = [[cond, new_cond_dict]]

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
                    
                    new_cond_dict = negative[0][1].copy() if negative and len(negative) > 0 else {}
                    new_cond_dict["pooled_output"] = pooled
                    final_negative_for_face = [[cond, new_cond_dict]]
            
            elif not clip and (exclusions or face_positive_prompt or face_negative_prompt):
                print("[Face Processor] Warning: Face prompts or exclusions were provided, but no CLIP model was connected. These options will be ignored.")

            face_masks = []
            
            np_masks = self.face_detector.detect_faces(
                image_tensor=processing_image, 
                enable_segmentation=enable_segmentation,
                detection_confidence=detection_confidence, 
                face_selection=face_selection
            )
            if np_masks:
                face_masks = [torch.from_numpy(m).unsqueeze(0) for m in np_masks]

            if not face_masks:
                return self.create_safe_fallback_outputs(input_image, processing_resolution)
            
            all_processed_faces, all_restore_info = [], []
            
            for i, face_mask in enumerate(face_masks):
                check_for_interruption()
                
                target_resolution = (processing_resolution, processing_resolution)

                cropped_face, sampler_mask_batch, restore_info = self.mask_processor.process_and_crop(
                    image_tensor=processing_image, 
                    mask_tensor=face_mask, 
                    crop_padding=crop_padding, 
                    processing_resolution=target_resolution, 
                    mask_expansion=mask_expansion,
                    enable_pre_upscale=enable_pre_upscale,
                    upscaler_model_name=upscaler_model,
                    upscaler_loader_callback=self.load_upscaler_model,
                    upscaler_run_callback=self.run_upscaler
                )
                
                if self.is_empty_detection(cropped_face, restore_info):
                    continue
                
                if manual_rotation != "None":
                    cropped_face_np = (cropped_face.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    rotated_face_np = self.apply_manual_rotation(cropped_face_np, manual_rotation)
                    cropped_face = torch.from_numpy(rotated_face_np.astype(np.float32) / 255.0).unsqueeze(0)
                    
                    sampler_mask_np = (sampler_mask_batch.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    rotated_mask_np = self.apply_manual_rotation(sampler_mask_np, manual_rotation)
                    sampler_mask_batch = torch.from_numpy(rotated_mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                    
                    restore_info['manual_rotation'] = manual_rotation
                else:
                    restore_info['manual_rotation'] = "None"
                
                processed_latent = self.run_inpaint_sampling(
                    cropped_face, sampler_mask_batch, model, vae, final_positive_for_face, final_negative_for_face,
                    steps, cfg_scale, sampler, scheduler, denoise_strength, seed + i,
                    sampling_mask_blur_size, sampling_mask_blur_strength,
                    enable_differential_diffusion
                )
                
                with torch.no_grad():
                    processed_face_batch = vae.decode(processed_latent["samples"])
                    if processed_face_batch.min() < -0.5: 
                        processed_face_batch = (processed_face_batch + 1.0) / 2.0
                    processed_face_batch = torch.clamp(processed_face_batch, 0.0, 1.0)
                
                if enable_lightness_rescue:
                    processed_face_batch = self.check_and_rescue_brightness(
                        cropped_face, processed_face_batch, sampler_mask_batch,
                        restore_info,
                        model, vae, final_positive_for_face, final_negative_for_face,
                        scheduler, sampler, seed + i, 
                        sampling_mask_blur_size, sampling_mask_blur_strength
                    )

                manual_rot = restore_info.get('manual_rotation', "None")

                if manual_rot != "None":
                    processed_face_np = (processed_face_batch.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    restored_face_np = self.reverse_manual_rotation(processed_face_np, manual_rot)
                    processed_face_batch = torch.from_numpy(restored_face_np.astype(np.float32) / 255.0).unsqueeze(0)
                    
                
                all_processed_faces.append(processed_face_batch)
                all_restore_info.append(restore_info)
                
                if len(face_masks) > 1 and i < len(face_masks) - 1 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not all_processed_faces:
                print("ERROR: All face processing failed. Returning original image.")
                return self.create_safe_fallback_outputs(original_image, processing_resolution)

            final_image = self.combine_all_faces_to_final_image(
                processing_image, all_processed_faces, all_restore_info, 
                blend_softness, enable_color_correction, 1.0
            )
            
            processed_face_output = self.create_combined_face_output(all_processed_faces, processing_resolution)
            side_by_side = self.create_unified_comparison(processing_image, all_processed_faces, all_restore_info, processing_resolution)
            final_mask = self.create_unified_mask(all_restore_info, processing_image)

            return (final_image, processed_face_output, side_by_side, final_mask)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"An error occurred during the main face processing workflow: {e}")
            return self.create_safe_fallback_outputs(input_image, processing_resolution)
    
    def apply_manual_rotation(self, image_np, rotation_option):
        if rotation_option == "None":
            return image_np
        elif rotation_option == "90° CW":
            return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_option == "90° CCW":
            return cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_option == "180°":
            return cv2.rotate(image_np, cv2.ROTATE_180)
        return image_np

    def reverse_manual_rotation(self, image_np, rotation_option):
        if rotation_option == "None":
            return image_np
        elif rotation_option == "90° CW":
            return cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_option == "90° CCW":
            return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_option == "180°":
            return cv2.rotate(image_np, cv2.ROTATE_180)
        return image_np
    
    

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
                        sampling_mask_blur_size, sampling_mask_blur_strength,
                        enable_differential_diffusion=True):
        try:
            work_model = model.clone()
            
            if enable_differential_diffusion:
                work_model.set_model_denoise_mask_function(self.differential_diffusion_function)

            result = self.process_single_face_sampling(
                cropped_face, face_mask, work_model, vae, positive, negative,
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
                if blur_size % 2 == 0:
                    blur_size += 1
                base_sigma = (blur_size - 1) / 8.0
                strength_tensor = torch.tensor(blur_strength - 1.0, device=device, dtype=torch.float32)
                multiplier = 1.0 + torch.tanh(strength_tensor) * 2.0
                actual_sigma = base_sigma * multiplier.item()

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
    
    def fast_upscale_bicubic(self, image_np_uint8, scale=4):
        try:
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0).unsqueeze(0)
            with torch.no_grad():
                upscaled_tensor = F.interpolate(
                    image_tensor.permute(0, 3, 1, 2),
                    scale_factor=scale,
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

    def fast_upscale_lanczos(self, image_np_uint8, scale=4):
        try:
            h, w = image_np_uint8.shape[:2]
            new_h, new_w = h * scale, w * scale
            
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