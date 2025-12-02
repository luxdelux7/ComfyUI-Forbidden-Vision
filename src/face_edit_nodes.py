import torch
import torch.nn.functional as F
import numpy as np
import cv2
import folder_paths
import comfy.model_management as model_management
import kornia

from .face_detector import ForbiddenVisionFaceDetector
from .mask_processor import ForbiddenVisionMaskProcessor
from .face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .utils import check_for_interruption, ensure_model_directories, get_ordered_upscaler_model_list, clean_model_name


class ForbiddenVisionFaceEditPrep:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()
        default_upscaler = "Fast 4x (Lanczos)"
        if "Fast 4x (Lanczos)" not in upscaler_models and upscaler_models:
            default_upscaler = upscaler_models[0]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for face crop and mask."}),
                "face_selection": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "0=All faces, 1=1st face, etc."}),
                "enable_segmentation": ("BOOLEAN", {"default": True, "tooltip": "Use AI segmentation. If disabled, creates oval masks."}),
                "detection_confidence": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "processing_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "crop_padding": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1}),
                "mask_expansion": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "enable_pre_upscale": ("BOOLEAN", {"default": True}),
                "upscaler_model": (upscaler_models, {"default": default_upscaler}),
                "isolate_face": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional external mask. If provided, detection is skipped."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FACE_INFO")
    RETURN_NAMES = ("cropped_face", "cropped_mask", "face_info")
    FUNCTION = "prepare_face"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        ensure_model_directories()
        self.face_detector = ForbiddenVisionFaceDetector()
        self.mask_processor = ForbiddenVisionMaskProcessor()
        self.upscaler_model = None
        self.upscaler_model_name = None

    def load_upscaler_model(self, model_name):
        clean_model_name_val = clean_model_name(model_name)

        if clean_model_name_val in [
            "Fast 4x (Bicubic AA)",
            "Fast 4x (Lanczos)",
            "Fast 2x (Bicubic AA)",
            "Fast 2x (Lanczos)"
        ]:
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

            from basicsr.utils import img2tensor, tensor2img
            from realesrgan import RealESRGANer

            device = model_management.get_torch_device()
            upsampler = RealESRGANer(
                model_path=model_path,
                scale=4,
                model=None,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=not model_management.is_device_mps(device),
                device=device,
            )

            self.upscaler_model = upsampler
            self.upscaler_model_name = clean_model_name_val
            print(f"Loaded upscaler model: {clean_model_name_val}")
            return True

        except Exception as e:
            print(f"Error loading upscaler model '{clean_model_name_val}': {e}")
            self.upscaler_model = None
            return False

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
            mask[:, :, :, :edge_size] = torch.minimum(
                mask[:, :, :, :edge_size],
                torch.linspace(0, 1, edge_size).view(1, 1, 1, edge_size)
            )
            mask[:, :, :, w-edge_size:] = torch.minimum(
                mask[:, :, :, w-edge_size:],
                torch.linspace(1, 0, edge_size).view(1, 1, 1, edge_size)
            )

            kernel_size = max(3, edge_size // 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = kernel_size / 3.0

            blurred = kornia.filters.gaussian_blur2d(image_tensor, (kernel_size, kernel_size), (sigma, sigma))
            cleaned = image_tensor * mask + blurred * (1 - mask)

            cleaned_np = cleaned.squeeze(0).permute(1, 2, 0).cpu().numpy()
            cleaned_np_uint8 = (np.clip(cleaned_np, 0, 1) * 255.0).round().astype(np.uint8)
            return cleaned_np_uint8
        except Exception as e:
            print(f"Error cleaning interpolation edges: {e}. Returning original image.")
            return image_np_uint8

    def run_upscaler(self, image_np_uint8):
        if self.upscaler_model_name in ["Fast 4x (Bicubic AA)", "Fast 2x (Bicubic AA)"]:
            scale = 4 if "4x" in self.upscaler_model_name else 2
            return self.fast_upscale_bicubic(image_np_uint8, scale=scale)

        if self.upscaler_model_name in ["Fast 4x (Lanczos)", "Fast 2x (Lanczos)"]:
            scale = 4 if "4x" in self.upscaler_model_name else 2
            return self.fast_upscale_lanczos(image_np_uint8, scale=scale)

        if self.upscaler_model is None:
            return image_np_uint8

        try:
            upsampler = self.upscaler_model
            output, _ = upsampler.enhance(image_np_uint8, outscale=1)
            return output
        except Exception as e:
            print(f"Error running upscaler '{self.upscaler_model_name}': {e}")
            return image_np_uint8

    def prepare_face(self, image, face_selection, enable_segmentation, detection_confidence,
                     processing_resolution, crop_padding, mask_expansion,
                     enable_pre_upscale, upscaler_model, isolate_face, mask=None):
        try:
            check_for_interruption()

            if image is None:
                h, w = 1024, 1024
                device = model_management.get_torch_device()
                empty_face = torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
                empty_info = {
                    "original_image": np.zeros((h, w, 3), dtype=np.uint8),
                    "original_image_size": (0, 0),
                    "crop_coords": (0, 0, 0, 0),
                    "face_bbox": (0, 0, 0, 0),
                    "target_size": (h, w),
                    "original_crop_size": (0, 0),
                    "blend_mask": np.zeros((h, w), dtype=np.float32),
                    "detection_angle": 0,
                }
                return (empty_face, empty_mask, empty_info)

            device = image.device
            h, w = image.shape[1], image.shape[2]

            if mask is not None:
                combined_mask = mask
            else:
                np_masks = self.face_detector.detect_faces(
                    image_tensor=image,
                    enable_segmentation=enable_segmentation,
                    detection_confidence=detection_confidence,
                    face_selection=face_selection
                )

                if not np_masks:
                    empty_face, empty_mask, empty_info = self.mask_processor.create_empty_outputs(
                        image_tensor=image.cpu(),
                        target_size=(processing_resolution, processing_resolution)
                    )
                    empty_face = empty_face.to(device)
                    empty_mask = empty_mask.to(device)
                    return (empty_face, empty_mask, empty_info)

                face_masks = [torch.from_numpy(m).unsqueeze(0) for m in np_masks]
                combined_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
                for mask_tensor in face_masks:
                    mask_tensor = mask_tensor.to(device)
                    combined_mask = torch.maximum(combined_mask, mask_tensor)

            target_resolution = (processing_resolution, processing_resolution)

            cropped_face, sampler_mask, restore_info = self.mask_processor.process_and_crop(
                image_tensor=image,
                mask_tensor=combined_mask,
                crop_padding=crop_padding,
                processing_resolution=target_resolution,
                mask_expansion=mask_expansion,
                enable_pre_upscale=enable_pre_upscale,
                upscaler_model_name=upscaler_model,
                upscaler_loader_callback=self.load_upscaler_model,
                upscaler_run_callback=self.run_upscaler
            )

            cropped_face = cropped_face.to(device)
            sampler_mask = sampler_mask.to(device)

            if isolate_face:
                mask_bchw = sampler_mask.unsqueeze(1)
                kh = max(3, (processing_resolution // 32) | 1)
                kw = kh
                sigma = max(1.0, kh / 3.0)
                blurred = kornia.filters.gaussian_blur2d(mask_bchw, (kh, kw), (sigma, sigma))
                blurred = torch.clamp(blurred, 0.0, 1.0)
                mask_bhwc = blurred.permute(0, 2, 3, 1)
                bg = torch.zeros_like(cropped_face)
                cropped_face = cropped_face * mask_bhwc + bg * (1.0 - mask_bhwc)

            return (cropped_face, sampler_mask, restore_info)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"[Face Edit Prep] Error preparing face for edit: {e}")
            try:
                device = image.device if image is not None else model_management.get_torch_device()
                res = processing_resolution
                empty_face = torch.zeros((1, res, res, 3), dtype=torch.float32, device=device)
                empty_mask = torch.zeros((1, res, res), dtype=torch.float32, device=device)
                empty_info = {
                    "original_image": np.zeros((res, res, 3), dtype=np.uint8),
                    "original_image_size": (0, 0),
                    "crop_coords": (0, 0, 0, 0),
                    "face_bbox": (0, 0, 0, 0),
                    "target_size": (res, res),
                    "original_crop_size": (0, 0),
                    "blend_mask": np.zeros((res, res), dtype=np.float32),
                    "detection_angle": 0,
                }
                return (empty_face, empty_mask, empty_info)
            except:
                device = model_management.get_torch_device()
                empty_face = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32, device=device)
                empty_mask = torch.zeros((1, 1024, 1024), dtype=torch.float32, device=device)
                empty_info = {
                    "original_image": np.zeros((1024, 1024, 3), dtype=np.uint8),
                    "original_image_size": (0, 0),
                    "crop_coords": (0, 0, 0, 0),
                    "face_bbox": (0, 0, 0, 0),
                    "target_size": (1024, 1024),
                    "original_crop_size": (0, 0),
                    "blend_mask": np.zeros((1024, 1024), dtype=np.float32),
                    "detection_angle": 0,
                }
                return (empty_face, empty_mask, empty_info)



class ForbiddenVisionFaceEditMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "Original full image."}),
                "edited_face": ("IMAGE", {"tooltip": "Edited face image at the same resolution as the cropped face from prep node."}),
                "face_info": ("FACE_INFO", {"tooltip": "Face info output from the prep node."}),
                "blend_softness": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "color_correction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "compositing_mask")
    FUNCTION = "merge_face"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        ensure_model_directories()
        self.processor = ForbiddenVisionFaceProcessorIntegrated()

    def merge_face(self, original_image, edited_face, face_info,
                   blend_softness, enable_color_correction, color_correction_strength):
        try:
            check_for_interruption()

            if original_image is None or edited_face is None or face_info is None:
                return (original_image, torch.zeros_like(original_image[:, :, :, 0]))

            if isinstance(face_info, (list, tuple)) and face_info and isinstance(face_info[0], dict):
                restore_info_list = [face_info[0]]
            elif isinstance(face_info, dict):
                restore_info_list = [face_info]
            else:
                return (original_image, torch.zeros_like(original_image[:, :, :, 0]))

            info = restore_info_list[0]
            crop_coords = info.get("crop_coords", (0, 0, 0, 0))
            ox1, oy1, ox2, oy2 = crop_coords
            if ox2 <= ox1 or oy2 <= oy1:
                mask = torch.zeros((original_image.shape[0], original_image.shape[1], original_image.shape[2]), device=original_image.device)
                return (original_image, mask)

            processed_faces = [edited_face]

            final_image = self.processor.combine_all_faces_to_final_image(
                original_image,
                processed_faces,
                restore_info_list,
                blend_softness,
                enable_color_correction,
                color_correction_strength
            )

            compositing_mask = self.processor.create_unified_mask(
                restore_info_list,
                original_image
            )

            return (final_image, compositing_mask)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"[Face Edit Merge] Error merging edited face: {e}")
            try:
                mask = torch.zeros((original_image.shape[0], original_image.shape[1], original_image.shape[2]), device=original_image.device)
                return (original_image, mask)
            except:
                device = model_management.get_torch_device()
                mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=device)
                return (None, mask)