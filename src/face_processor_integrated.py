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
from .face_detector import ForbiddenVisionFaceDetector
from .mask_processor import ForbiddenVisionMaskProcessor
from .utils import check_for_interruption, get_yolo_models, get_sam_models, get_ordered_upscaler_model_list

class ForbiddenVisionFaceProcessorIntegrated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
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
                
                "processing_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "The resolution at which the cropped mask area will be processed."}),
                "enable_pre_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Enable to upscale small faces with an AI model before processing."}),
                "upscaler_model": (get_ordered_upscaler_model_list(), {"tooltip": "The AI model used for pre-upscaling small faces. Only active if 'enable_pre_upscale' is on."}),
                "crop_padding": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Padding added to the mask's bounding box before inpaint."}),
                
                "face_positive_prompt": ("STRING", {"multiline": True, "default": "face focus"}),
                "face_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "replace_positive_prompt": ("BOOLEAN", {"default": False}),
                "replace_negative_prompt": ("BOOLEAN", {"default": False}),
                
                "blend_softness": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "sampling_mask_blur_size": ("INT", {"default": 21, "min": 1, "max": 101, "step": 2}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.1}),
                "enable_vertical_flip": ("BOOLEAN", {"default": False}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "color_correction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                "face_selection": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "0=All faces, 1=1st face, etc. Used by both mask input and internal detector."}),
                "detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold for the internal BBox detector (YOLO)."}),
                "sam_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold for the internal SAM mask refinement."}),
                "bbox_model": (get_yolo_models(), {"tooltip": "Internal BBOX detector model (YOLO). Used if 'mask' is not connected."}),
                "sam_model": (get_sam_models(), {"tooltip": "Internal Segment Anything Model (SAM). Used if 'mask' is not connected."}),
                "attempt_face_completion": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Attempt to find and graft separated parts of the face (e.g., chin obscured by hair)."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional: Face mask for processing. If connected, internal detection is skipped."}),
                "clip": ("CLIP", {"tooltip": "Optional: Required only if using face prompts."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "processed_face", "side_by_side_comparison", "final_mask")
    FUNCTION = "process_face_complete"
    CATEGORY = "Forbidden Vision"
    @staticmethod
    def get_ordered_upscaler_models():
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
        for preferred in preferred_models:
            for model in all_models:
                if preferred in model or model in preferred:
                    if model not in found_preferred:
                        found_preferred.append(model)
                    break
        
        remaining_models = [model for model in all_models if model not in found_preferred]
        
        ordered_models = found_preferred + sorted(remaining_models)
        
        return ordered_models if ordered_models else ["None Found"]
    
    
    def __init__(self):
        self.face_detector = ForbiddenVisionFaceDetector()
        self.mask_processor = ForbiddenVisionMaskProcessor()
        self.last_sampling_key = None
        self.last_sampling_result = None
        self.upscaler_model = None
        self.upscaler_model_name = None

    def load_upscaler_model(self, model_name):
        if self.upscaler_model is not None and self.upscaler_model_name == model_name:
            return True

        try:
            model_path = folder_paths.get_full_path("upscale_models", model_name)
            if model_path is None:
                print(f"Upscaler model '{model_name}' not found.")
                return False
            
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            
            self.upscaler_model = upscaler_loader.load_model(model_name)[0]
            self.upscaler_model_name = model_name
            
            return True
            
        except model_management.InterruptProcessingException:
            raise
        except KeyError:
            print("Error: Could not find 'UpscaleModelLoader' in nodes.NODE_CLASS_MAPPINGS.")
            return False
        except Exception as e:
            print(f"An error occurred loading upscaler model {model_name}: {e}")
            self.upscaler_model = None
            self.upscaler_model_name = None
            return False

    def run_upscaler(self, image_np_uint8):
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
            
            # Ensure tensors are in the correct format (B, C, H, W) and float32
            processed_face_permuted = processed_face_tensor.permute(0, 3, 1, 2).float()
            original_crop_permuted = original_crop_tensor.permute(0, 3, 1, 2).float()

            # Resize original crop to match processed face size
            original_face_resized = F.interpolate(
                original_crop_permuted, 
                size=processed_face_permuted.shape[2:], 
                mode='bicubic', 
                align_corners=False, 
                antialias=True
            )

            # Convert to LAB color space on GPU
            processed_lab = kornia.color.rgb_to_lab(processed_face_permuted)
            target_lab = kornia.color.rgb_to_lab(original_face_resized)

            # Calculate stats on GPU
            source_mean = torch.mean(processed_lab, dim=(2, 3), keepdim=True)
            source_std = torch.std(processed_lab, dim=(2, 3), keepdim=True)
            target_mean = torch.mean(target_lab, dim=(2, 3), keepdim=True)
            target_std = torch.std(target_lab, dim=(2, 3), keepdim=True)
            
            # Apply color transfer on GPU
            result_lab = (processed_lab - source_mean) * (target_std / (source_std + 1e-6)) + target_mean
            
            # Convert back to RGB on GPU
            corrected_face_permuted = kornia.color.lab_to_rgb(result_lab)

            # Blend with original based on strength
            if correction_strength < 1.0:
                corrected_face_permuted = processed_face_permuted * (1.0 - correction_strength) + corrected_face_permuted * correction_strength
            
            # Clamp and permute back to (B, H, W, C)
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
            
            # Ensure mask is a 4D tensor (B, C, H, W) for Kornia
            mask_4d = clean_sam_mask_tensor.unsqueeze(0)
            
            core_mask = (mask_4d > 0.5).float()
            
            # Calculate dynamic kernel sizes on GPU
            mask_size = max(w, h, 1)
            blend_ratio = blend_softness / 400.0
            
            # Dilation for blur expansion
            expand_amount = int(mask_size * blend_ratio * 0.8)
            if expand_amount > 0:
                expand_kernel_size = expand_amount * 2 + 1
                expand_kernel = torch.ones(expand_kernel_size, expand_kernel_size, device=mask_4d.device)
                expanded_for_blur = kornia.morphology.dilation(core_mask, expand_kernel)
            else:
                expanded_for_blur = core_mask

            # Gaussian blur for feathering
            blur_amount = int(mask_size * blend_ratio * 1.2)
            blur_kernel_size = max(3, blur_amount * 2 + 1)
            sigma = blur_kernel_size / 3.0 # Approximate sigma
            feathering_zone = kornia.filters.gaussian_blur2d(
                expanded_for_blur, (blur_kernel_size, blur_kernel_size), (sigma, sigma)
            )

            # Combine core mask and feathering zone
            final_mask = torch.max(core_mask, feathering_zone)

            # Edge Fading
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

            # Start with the original image tensor on the GPU
            final_canvas_tensor = original_image_tensor.clone()
            
            # Create a unified mask for compositing on the GPU
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

                # Ensure processed face is on the correct device
                processed_face_tensor = processed_face_tensor.to(device)

                # Perform color correction on GPU
                if enable_color_correction and color_correction_strength > 0.0:
                    original_crop_tensor = original_image_tensor[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
                    processed_face_tensor = self._perform_color_correction_gpu(
                        processed_face_tensor, original_crop_tensor, color_correction_strength
                    )

                # Resize processed face on GPU
                resized_processed_face = F.interpolate(
                    processed_face_tensor.permute(0, 3, 1, 2), # B, C, H, W
                    size=(actual_crop_h, actual_crop_w),
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1) # B, H, W, C
                
                # Place the resized face onto a temporary canvas
                temp_face_canvas = torch.zeros_like(final_canvas_tensor)
                temp_face_canvas[:, crop_y1:crop_y2, crop_x1:crop_x2, :] = resized_processed_face

                # Create the blend mask for this specific face on GPU
                full_blend_mask = torch.from_numpy(restore_info.get("blend_mask")).to(device)
                if full_blend_mask is None:
                    paste_mask_crop = torch.ones((actual_crop_h, actual_crop_w), device=device)
                else:
                    paste_mask_crop = full_blend_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                compositing_mask_crop = self.create_compositing_blend_mask_gpu(paste_mask_crop.unsqueeze(0), blend_softness)
                
                # Place the crop's mask onto a full-size mask canvas
                temp_mask_canvas = torch.zeros_like(unified_compositing_mask)
                temp_mask_canvas[:, crop_y1:crop_y2, crop_x1:crop_x2] = compositing_mask_crop
                
                # Blend the temporary face canvas with the main canvas using the temporary mask
                final_canvas_tensor = temp_face_canvas * temp_mask_canvas.unsqueeze(-1) + final_canvas_tensor * (1.0 - temp_mask_canvas.unsqueeze(-1))

            check_for_interruption()
            
            return final_canvas_tensor

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error combining faces to final image on GPU: {e}")
            return original_image_tensor
    def process_face_complete(self, image, model, vae, positive, negative, 
                            steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                            processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                            face_positive_prompt, face_negative_prompt, replace_positive_prompt, replace_negative_prompt,
                            blend_softness, mask_expansion,
                            sampling_mask_blur_size, sampling_mask_blur_strength,
                            enable_vertical_flip, enable_color_correction, color_correction_strength,
                            face_selection, detection_confidence, sam_threshold, bbox_model, sam_model, attempt_face_completion,
                            mask=None, clip=None):
        try:
            check_for_interruption()
            
            if image is None or model is None or vae is None:
                print("ERROR: Required inputs (image, model, vae) are missing.")
                return self.create_safe_fallback_outputs(image, processing_resolution)

            original_image = image
            face_masks = []


            final_positive, final_negative = self.enhance_conditioning_with_face_prompts(
                positive, negative, face_positive_prompt, face_negative_prompt, 
                replace_positive_prompt, replace_negative_prompt, clip
            )

            if mask is not None:
                face_masks = [mask]
            else:
                np_masks = self.face_detector.detect_faces(
                    image_tensor=image, 
                    bbox_model_name=bbox_model, 
                    sam_model_name=sam_model,
                    detection_confidence=detection_confidence, 
                    sam_threshold=sam_threshold, 
                    face_selection=face_selection,
                    attempt_face_completion=attempt_face_completion
                )
                if np_masks:
                    face_masks = [torch.from_numpy(m).unsqueeze(0) for m in np_masks]

            if not face_masks:
                print("ERROR: No face masks found. Cannot proceed.")
                return self.create_safe_fallback_outputs(image, processing_resolution)

            current_sampling_key = self.get_sampling_cache_key_only(
                image, mask, model, vae, final_positive, final_negative,
                steps, cfg_scale, sampler, scheduler, 
                denoise_strength, seed, processing_resolution, enable_pre_upscale, upscaler_model, crop_padding, 
                mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength, 
                enable_vertical_flip, face_selection, detection_confidence, sam_threshold, bbox_model, sam_model,
                attempt_face_completion
            )

            if self.last_sampling_key == current_sampling_key and self.last_sampling_result is not None:
                all_processed_faces, all_restore_info = self.last_sampling_result
            else:
                
                processing_image = torch.flip(image, dims=[1]) if enable_vertical_flip else image.clone()
                if enable_vertical_flip and mask is not None:
                    face_masks = [torch.flip(m, dims=[1]) for m in face_masks]
                
                all_processed_faces, all_restore_info = [], []
                
                for i, face_mask in enumerate(face_masks):
                    check_for_interruption()
                    processed_result = self.process_single_face_unified(
                        processing_image, face_mask, model, vae, final_positive, final_negative,
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed + i,
                        processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                        mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength
                    )
                    if processed_result:
                        _, processed_face, restore_info = processed_result
                        all_processed_faces.append(processed_face)
                        all_restore_info.append(restore_info)
                        if len(face_masks) > 1 and i < len(face_masks) - 1 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                self.last_sampling_key = current_sampling_key
                self.last_sampling_result = (all_processed_faces, all_restore_info)

            if not all_processed_faces:
                print("ERROR: All face processing failed. Returning original image.")
                return self.create_safe_fallback_outputs(original_image, processing_resolution)


            compositing_base_image = torch.flip(image, dims=[1]) if enable_vertical_flip else original_image.clone()
            final_image = self.combine_all_faces_to_final_image(
                compositing_base_image, all_processed_faces, all_restore_info, 
                blend_softness, enable_color_correction, color_correction_strength
            )
            

            processed_face_output = self.create_combined_face_output(all_processed_faces, processing_resolution)
            side_by_side = self.create_unified_comparison(original_image, all_processed_faces, all_restore_info, processing_resolution)
            final_mask = self.create_unified_mask(all_restore_info, original_image)
            
            if enable_vertical_flip:
                processed_face_output = torch.flip(processed_face_output, dims=[1])
                side_by_side = torch.flip(side_by_side, dims=[1])
                final_mask = torch.flip(final_mask, dims=[1])

            return (final_image, processed_face_output, side_by_side, final_mask)
            
        except model_management.InterruptProcessingException:
            print("Face processing cancelled by user.")
            raise
        except Exception as e:
            print(f"An error occurred during the main face processing workflow: {e}")
            (fallback_image, fallback_processed, fallback_comparison, fallback_mask) = self.create_safe_fallback_outputs(original_image, processing_resolution)
            if enable_vertical_flip:
                fallback_image = torch.flip(fallback_image, dims=[1])
                fallback_processed = torch.flip(fallback_processed, dims=[1])
                fallback_comparison = torch.flip(fallback_comparison, dims=[1])
                fallback_mask = torch.flip(fallback_mask, dims=[1])
            return (fallback_image, fallback_processed, fallback_comparison, fallback_mask)
    
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
                                detection_confidence, sam_threshold, bbox_model, sam_model, attempt_face_completion):
        try:
            image_sample = image[0, ::96, ::96, :].flatten()[:50] if image.numel() > 2500 else image.flatten()[:50]
            image_hash = hash(tuple(image_sample.cpu().numpy().round(3)))
            
            mask_hash = 0
            if mask is not None:
                mask_sample = mask[0, ::96, ::96].flatten()[:50] if mask.numel() > 2500 else mask.flatten()[:50]
                mask_hash = hash(tuple(mask_sample.cpu().numpy().round(3)))
            else:
                mask_hash = hash((detection_confidence, sam_threshold, bbox_model, sam_model))

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
                enable_vertical_flip, face_selection, attempt_face_completion
            )
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error generating sampling cache key: {e}")
            return None
    
    def process_single_face_unified(self, image, mask, model, vae, final_positive, final_negative,
                        steps, cfg_scale, sampler, scheduler, denoise_strength, seed,
                        processing_resolution, enable_pre_upscale, upscaler_model, crop_padding,
                        mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength):
        try:
            if mask is None:
                return None
            
            cropped_face, sampler_mask_batch, restore_info = self.mask_processor.process_and_crop(
                image_tensor=image, 
                mask_tensor=mask, 
                crop_padding=crop_padding, 
                processing_resolution=processing_resolution, 
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
                # Permute to (B, C, H, W) for interpolate
                face_permuted = face_tensor.to(device).permute(0, 3, 1, 2)
                resized_face = F.interpolate(
                    face_permuted, 
                    size=(processing_resolution, processing_resolution), 
                    mode='bicubic', 
                    align_corners=False,
                    antialias=True
                )
                # Permute back to (B, H, W, C)
                face_tensors.append(resized_face.permute(0, 2, 3, 1))
            
            if len(face_tensors) == 2:
                combined = torch.cat(face_tensors, dim=2) # Concatenate along width
            else:
                rows = []
                for i in range(0, len(face_tensors), 2):
                    row_faces = face_tensors[i:i+2]
                    if len(row_faces) == 1:
                        # Add a black tensor if there's an odd number of faces
                        black_tensor = torch.zeros_like(row_faces[0])
                        row_faces.append(black_tensor)
                    row = torch.cat(row_faces, dim=2) # Concatenate along width
                    rows.append(row)
                combined = torch.cat(rows, dim=1) # Concatenate rows along height
            
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
                
                # Slice the original image on the GPU
                original_crop = original_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
                
                # Resize original crop on GPU
                original_resized = F.interpolate(
                    original_crop.permute(0, 3, 1, 2),
                    size=(processing_resolution, processing_resolution),
                    mode='bicubic', align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)

                # Resize processed face on GPU
                processed_resized = F.interpolate(
                    processed_face.to(device).permute(0, 3, 1, 2),
                    size=(processing_resolution, processing_resolution),
                    mode='bicubic', align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)
                
                # Concatenate side-by-side on GPU
                face_comparison = torch.cat([original_resized, processed_resized], dim=2) # Cat along width
                comparisons.append(face_comparison)
            
            if not comparisons:
                return original_image
            
            # Stack all comparisons vertically on GPU
            final_comparison = torch.cat(comparisons, dim=1) # Cat along height
            
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
                    # Transfer mask to GPU once and combine using torch.maximum
                    blend_mask_tensor = torch.from_numpy(blend_mask_np).to(device)
                    combined_mask = torch.maximum(combined_mask, blend_mask_tensor)

            # Move the single combined mask to CPU for the CPU-only polishing step
            combined_mask_np = combined_mask.squeeze(0).cpu().numpy()
            
            # Polish on CPU
            polished_mask_np = self.mask_processor.polish_mask(combined_mask_np)
            
            # Return the final polished mask as a tensor on the correct device
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
                # Create side-by-side comparison on GPU
                empty_comparison = torch.cat([image, black_side], dim=2)
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
            else:
                # Default sizes if no image is provided
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
            # Ensure the comparison image has the correct double-width shape
            fallback_comp = torch.zeros((1, 512, 1024, 3), dtype=torch.float32, device=device)
            return (fallback_img, fallback_img, fallback_comp, fallback_mask)
    
    def enhance_conditioning_with_face_prompts(self, positive, negative, face_positive_prompt, face_negative_prompt, replace_positive_prompt, replace_negative_prompt, clip):
        try:
            has_positive_prompt = face_positive_prompt and face_positive_prompt.strip() != ""
            has_negative_prompt = face_negative_prompt and face_negative_prompt.strip() != ""
            
            if not has_positive_prompt and not has_negative_prompt:
                return positive, negative
            
            if clip is None:
                return positive, negative
            
            enhanced_positive = self.modify_conditioning_with_prompt(
                positive, face_positive_prompt, "positive", clip, replace_positive_prompt
            ) if has_positive_prompt else positive
            
            enhanced_negative = self.modify_conditioning_with_prompt(
                negative, face_negative_prompt, "negative", clip, replace_negative_prompt
            ) if has_negative_prompt else negative
            
            return enhanced_positive, enhanced_negative
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error enhancing conditioning with face prompts: {e}")
            return positive, negative

    def modify_conditioning_with_prompt(self, conditioning, face_prompt, prompt_type, clip, overwrite_mode):
        try:
            if not face_prompt or not face_prompt.strip():
                return conditioning
            
            tokens = clip.tokenize(face_prompt.strip())
            new_cond, new_pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            if overwrite_mode:
                return [[new_cond, {"pooled_output": new_pooled}]]
            else:
                if not conditioning:
                    return [[new_cond, {"pooled_output": new_pooled}]]
                
                new_opts = conditioning[0][1].copy()
                new_opts["pooled_output"] = new_pooled
                new_item = [new_cond, new_opts]
                
                return [new_item] + conditioning

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error modifying {prompt_type} conditioning: {e}")
            return conditioning

    
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
            
            # Reshape and resize the mask on the GPU.
            # We keep it as a 4D tensor (B, C, H, W) as Kornia prefers this format.
            mask_4d = face_mask.reshape(1, 1, face_mask.shape[-2], face_mask.shape[-1])
            resized_mask_4d = F.interpolate(mask_4d, size=(latent_height, latent_width), mode='bilinear', align_corners=False)

            if blur_size > 1 and blur_strength > 0:
                # Use standard math library for scalar calculations. It's fast and avoids numpy.
                import math
                
                # Ensure blur_size is odd
                if blur_size % 2 == 0:
                    blur_size += 1
                
                # Calculate sigma using standard math, not numpy
                base_sigma = (blur_size - 1) / 8.0
                actual_sigma = base_sigma * (1 + math.tanh(blur_strength - 1) * 2)
                
                # Apply Gaussian blur on the GPU using Kornia
                blurred_mask_4d = kornia.filters.gaussian_blur2d(
                    resized_mask_4d, (blur_size, blur_size), (actual_sigma, actual_sigma)
                )
                
                # Return the final blurred mask as a 3D tensor (B, H, W) for the KSampler
                return blurred_mask_4d.squeeze(1)

            # If no blur is applied, just return the resized mask as a 3D tensor
            return resized_mask_4d.squeeze(1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error processing inpaint mask on GPU: {e}")
            # A safe fallback in case of an error during processing
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