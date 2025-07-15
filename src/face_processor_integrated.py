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
from .face_detector import ForbiddenVisionFaceDetector
from .mask_processor import ForbiddenVisionMaskProcessor
from .utils import safe_tensor_to_numpy, safe_numpy_to_tensor, check_for_interruption, get_yolo_models, get_sam_models

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
                "upscaler_model": (ForbiddenVisionFaceProcessorIntegrated.get_ordered_upscaler_models(), {"tooltip": "The AI model used for pre-upscaling small faces. Only active if 'enable_pre_upscale' is on."}),
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
    
    def combine_all_faces_to_final_image(self, original_image, all_processed_faces, all_restore_info, blend_softness, enable_color_correction=False, color_correction_strength=1.0):
        try:
            check_for_interruption()
            
            if not all_processed_faces or not all_restore_info:
                return original_image

            original_np = self.fast_tensor_to_numpy(original_image.clone(), target_range=(0, 1))
            h, w = original_np.shape[:2]

            new_faces_canvas = original_np.copy()
            unified_compositing_mask = np.zeros((h, w), dtype=np.float32)

            for processed_face_tensor, restore_info in zip(all_processed_faces, all_restore_info):
                check_for_interruption()
                
                if isinstance(restore_info, list):
                    restore_info = restore_info[0]

                crop_coords = restore_info["crop_coords"]
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                actual_crop_w, actual_crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

                if actual_crop_w <= 0 or actual_crop_h <= 0:
                    continue

                processed_face_np = self.fast_tensor_to_numpy(processed_face_tensor, target_range=(0, 1))
                
                if enable_color_correction and color_correction_strength > 0.0:
                    original_crop_np = original_np[crop_y1:crop_y2, crop_x1:crop_x2]

                    processed_face_np = self._perform_color_correction(
                        processed_face_np, original_crop_np, color_correction_strength
                    )

                resized_processed_face = self.high_quality_resize(
                    processed_face_np, (actual_crop_w, actual_crop_h)
                )
                
                new_faces_canvas[crop_y1:crop_y2, crop_x1:crop_x2] = resized_processed_face

                full_blend_mask = restore_info.get("blend_mask")
                if full_blend_mask is None:
                    paste_mask_crop = np.ones((actual_crop_h, actual_crop_w), dtype=np.float32)
                else:
                    paste_mask_crop = full_blend_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                compositing_mask_crop = self.create_compositing_blend_mask(paste_mask_crop, blend_softness)
                
                unified_compositing_mask[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(
                    unified_compositing_mask[crop_y1:crop_y2, crop_x1:crop_x2],
                    compositing_mask_crop
                )

            check_for_interruption()

            unified_compositing_mask_3d = np.stack([unified_compositing_mask] * 3, axis=-1)
            
            final_result_np = (new_faces_canvas * unified_compositing_mask_3d) + (original_np * (1.0 - unified_compositing_mask_3d))
            
            final_tensor = self.fast_numpy_to_tensor(final_result_np, input_range=(0, 1))
            
            return final_tensor

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error combining faces to final image: {e}")
            return original_image
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
            if not processed_faces:
                return torch.zeros((1, processing_resolution, processing_resolution, 3))
            
            if len(processed_faces) == 1:
                return processed_faces[0]
            
            face_crops = []
            for processed_face in processed_faces:
                processed_np = self.fast_tensor_to_numpy(processed_face, target_range=(0, 255))
                processed_resized = cv2.resize(processed_np, (processing_resolution, processing_resolution), interpolation=cv2.INTER_CUBIC)
                face_crops.append(processed_resized)
            
            if len(face_crops) == 2:
                combined = np.concatenate(face_crops, axis=1)
            else:
                rows = []
                for i in range(0, len(face_crops), 2):
                    row_faces = face_crops[i:i+2]
                    if len(row_faces) == 1:
                        row_faces.append(np.zeros_like(row_faces[0]))
                    row = np.concatenate(row_faces, axis=1)
                    rows.append(row)
                combined = np.concatenate(rows, axis=0)
            
            return self.fast_numpy_to_tensor(combined, input_range=(0, 255))
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating combined face output: {e}")
            return processed_faces[0] if processed_faces else torch.zeros((1, processing_resolution, processing_resolution, 3))

    def create_unified_comparison(self, original_image, processed_faces, restore_info_list, processing_resolution):
        try:
            if not processed_faces or not restore_info_list:
                return original_image
            
            original_np = self.fast_tensor_to_numpy(original_image, target_range=(0, 255))
            comparisons = []
            
            for processed_face, restore_info in zip(processed_faces, restore_info_list):
                crop_coords = restore_info["crop_coords"]
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue
                
                original_crop = original_np[crop_y1:crop_y2, crop_x1:crop_x2]
                original_resized = cv2.resize(original_crop, (processing_resolution, processing_resolution), interpolation=cv2.INTER_CUBIC)
                
                processed_np = self.fast_tensor_to_numpy(processed_face, target_range=(0, 255))
                processed_resized = cv2.resize(processed_np, (processing_resolution, processing_resolution), interpolation=cv2.INTER_CUBIC)
                
                face_comparison = np.concatenate([original_resized, processed_resized], axis=1)
                comparisons.append(face_comparison)
            
            if not comparisons:
                return original_image
            
            if len(comparisons) == 1:
                final_comparison = comparisons[0]
            else:
                final_comparison = np.concatenate(comparisons, axis=0)
            
            return self.fast_numpy_to_tensor(final_comparison, input_range=(0, 255))
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating unified comparison: {e}")
            return original_image

    def create_unified_mask(self, restore_info_list, image):
        try:
            if not restore_info_list:
                image_shape = image.shape
                return torch.zeros((1, image_shape[1], image_shape[2]), dtype=torch.float32)
            
            
            image_shape = image.shape
            original_height, original_width = image_shape[1], image_shape[2]
            combined_mask = np.zeros((original_height, original_width), dtype=np.float32)
            
            for restore_info in restore_info_list:
                blend_mask = restore_info.get("blend_mask")
                if blend_mask is not None:
                    combined_mask = np.maximum(combined_mask, blend_mask)

            polished_mask = self.mask_processor.polish_mask(combined_mask)
            
            return torch.from_numpy(polished_mask).unsqueeze(0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating unified mask: {e}")
            image_shape = image.shape
            return torch.zeros((1, image_shape[1], image_shape[2]), dtype=torch.float32)
        
    
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
            final_output = image.clone() if image is not None else torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            empty_processed = torch.zeros((1, processing_resolution, processing_resolution, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)

            if image is not None:
                original_np = self.tensor_to_numpy_comparison(image)
                h, w = original_np.shape[:2]
                black_side = np.zeros_like(original_np)
                comparison_np = np.concatenate([original_np, black_side], axis=1)
                empty_comparison = self.numpy_to_tensor_comparison(comparison_np)
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
            else:
                empty_comparison = torch.zeros((1, 512, 1024, 3), dtype=torch.float32)
            
            return (final_output, empty_processed, empty_comparison, empty_mask)
        except model_management.InterruptProcessingException:
            raise
        except:
            fallback_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            fallback_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (fallback_img, fallback_img, fallback_img, fallback_mask)

    
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

    def tensor_to_numpy_comparison(self, tensor):
        return self.fast_tensor_to_numpy(tensor, target_range=(0, 255))

    def numpy_to_tensor_comparison(self, np_array):
        return self.fast_numpy_to_tensor(np_array, input_range=(0, 255))
    
    def _perform_color_correction(self, processed_face_np, original_crop_np, correction_strength):
        try:
            original_face_resized = cv2.resize(original_crop_np,
                                            (processed_face_np.shape[1], processed_face_np.shape[0]),
                                            interpolation=cv2.INTER_CUBIC)

            corrected_face = self.lab_color_transfer(processed_face_np, original_face_resized)

            if correction_strength < 1.0:
                corrected_face = processed_face_np * (1.0 - correction_strength) + corrected_face * correction_strength

            return np.clip(corrected_face, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in _perform_color_correction helper: {e}")
            return processed_face_np
        
    def lab_color_transfer(self, source_image, target_image):
        try:
            if source_image.shape != target_image.shape:
                target_image = cv2.resize(target_image, (source_image.shape[1], source_image.shape[0]))
            
            source_uint8 = (np.clip(source_image, 0, 1) * 255).astype(np.uint8)
            target_uint8 = (np.clip(target_image, 0, 1) * 255).astype(np.uint8)
            
            source_lab = cv2.cvtColor(source_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
            source_std = np.std(source_lab.reshape(-1, 3), axis=0)
            
            target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
            target_std = np.std(target_lab.reshape(-1, 3), axis=0)
            
            result_lab = source_lab.copy()
            
            for i in range(3):
                if source_std[i] > 1e-6:
                    result_lab[:, :, i] = (result_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
            
            result_lab = np.clip(result_lab, 0, 255)
            result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            return result_rgb.astype(np.float32) / 255.0
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in LAB color transfer: {e}")
            return source_image

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
            mask_tensor = face_mask.reshape(1, 1, face_mask.shape[-2], face_mask.shape[-1])
            resized_mask = torch.nn.functional.interpolate(mask_tensor, size=(latent_height, latent_width), mode='bilinear', align_corners=False).squeeze(1)

            resized_mask_np = resized_mask.cpu().numpy().squeeze()
            
            if blur_size > 1 and blur_strength > 0:
                base_sigma = (blur_size - 1) / 8.0
                actual_sigma = base_sigma * (1 + np.tanh(blur_strength - 1) * 2)
                
                if blur_size % 2 == 0:
                    blur_size += 1
                
                blurred_mask = cv2.GaussianBlur(resized_mask_np, (blur_size, blur_size), actual_sigma)
                resized_mask = torch.from_numpy(blurred_mask).unsqueeze(0).to(device)

            return resized_mask
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error processing inpaint mask: {e}")
            return resized_mask
   

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

    def high_quality_resize(self, image_float, target_size):
        try:
            target_w, target_h = target_size
            current_h, current_w = image_float.shape[:2]
            
            if target_w <= 0 or target_h <= 0: 
                return image_float
            if current_w == target_w and current_h == target_h:
                return image_float
                
            if image_float.dtype != np.float32: 
                image_float = image_float.astype(np.float32)
            
            scale_factor = min(target_w / current_w, target_h / current_h)
            
            if scale_factor > 1.0:
                return cv2.resize(image_float, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif scale_factor > 0.75:
                return cv2.resize(image_float, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                resized = cv2.resize(image_float, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if scale_factor < 0.5:
                    amount = 0.3
                    sigma = 0.6
                    blurred = cv2.GaussianBlur(resized, (0, 0), sigma)
                    detail = resized - blurred
                    sharpened = resized + detail * amount
                    return np.clip(sharpened, 0.0, 1.0)
                return resized
                
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in high quality resize: {e}")
            return cv2.resize(image_float, target_size, interpolation=cv2.INTER_AREA)

    def create_compositing_blend_mask(self, clean_sam_mask, blend_softness):
        try:
            if blend_softness <= 0:
                return (clean_sam_mask > 0.5).astype(np.float32)
            h, w = clean_sam_mask.shape[:2]
            core_mask = (clean_sam_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return np.zeros_like(clean_sam_mask, dtype=np.float32)
            
            largest_contour = max(contours, key=cv2.contourArea)
            _, _, mask_w, mask_h = cv2.boundingRect(largest_contour)
            mask_size = max(mask_w, mask_h, 1)
            blend_ratio = blend_softness / 400.0
            expand_amount = int(mask_size * blend_ratio * 0.8)
            if expand_amount > 0:
                expand_kernel_size = expand_amount * 2 + 1
                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_kernel_size, expand_kernel_size))
                expanded_for_blur = cv2.dilate(core_mask, expand_kernel, iterations=1)
            else:
                expanded_for_blur = core_mask.copy()
            blur_amount = int(mask_size * blend_ratio * 1.2)
            blur_kernel_size = max(3, blur_amount * 2 + 1)
            
            feathering_zone = cv2.GaussianBlur(expanded_for_blur, (blur_kernel_size, blur_kernel_size), 0)
            final_mask_uint8 = np.maximum(core_mask, feathering_zone)
            final_mask_float = final_mask_uint8.astype(np.float32) / 255.0
            fade_pixels = int(min(h, w) * 0.02)
            if fade_pixels > 0:
                for i in range(fade_pixels):
                    fade_factor = i / float(fade_pixels)
                    final_mask_float[i, :] *= fade_factor
                    final_mask_float[h-1-i, :] *= fade_factor
                    final_mask_float[:, i] *= fade_factor
                    final_mask_float[:, w-1-i] *= fade_factor
            
            final_mask_float[0, :], final_mask_float[-1, :], final_mask_float[:, 0], final_mask_float[:, -1] = 0, 0, 0, 0
            return np.clip(final_mask_float, 0.0, 1.0)
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating hybrid blend mask: {e}")
            return (clean_sam_mask > 0.5).astype(np.float32)
        
    def fast_tensor_to_numpy(self, tensor, target_range=(0, 255)):
        try:
            if tensor.device != torch.device('cpu'):
                tensor = tensor.cpu()
            
            np_array = tensor.squeeze().numpy()
            
            if target_range == (0, 255):
                if np_array.max() <= 1.0:
                    return (np_array * 255.0).astype(np.uint8)
                else:
                    return np.clip(np_array, 0, 255).astype(np.uint8)
            else:
                if np_array.max() > 1.0:
                    return np_array / 255.0
                else:
                    return np_array
                    
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in fast tensor conversion: {e}")
            return safe_tensor_to_numpy(tensor, target_range)

    def fast_numpy_to_tensor(self, np_array, input_range=(0, 255)):
        try:
            if input_range == (0, 255):
                if np_array.dtype == np.uint8:
                    tensor_data = np_array.astype(np.float32) / 255.0
                else:
                    tensor_data = np.clip(np_array / 255.0, 0.0, 1.0)
            else:
                tensor_data = np.clip(np_array, 0.0, 1.0).astype(np.float32)
            
            return torch.from_numpy(tensor_data).unsqueeze(0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in fast numpy conversion: {e}")
            return safe_numpy_to_tensor(np_array, input_range)