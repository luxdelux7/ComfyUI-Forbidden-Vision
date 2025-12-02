import torch
import numpy as np
from .face_detector import ForbiddenVisionFaceDetector
from .mask_processor import ForbiddenVisionMaskProcessor
from .utils import check_for_interruption
import cv2

class ForbiddenVisionFacePrepForEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_selection": ("INT", {"default": 1, "min": 1, "max": 5}),
                "enable_segmentation": ("BOOLEAN", {"default": True}),
                "detection_confidence": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
                "processing_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "crop_padding": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.0}),
                "mask_expansion": ("INT", {"default": 12, "min": 0, "max": 80}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FACE_INFO")
    RETURN_NAMES = ("cropped_face", "face_mask", "face_info")
    FUNCTION = "execute"
    CATEGORY = "Forbidden Vision"

    def execute(self, image, face_selection, enable_segmentation,
                detection_confidence, processing_resolution,
                crop_padding, mask_expansion, mask_blur):

        check_for_interruption()

        img_tensor = image
        detector = ForbiddenVisionFaceDetector()
        masks = detector.detect_faces(
            img_tensor,
            enable_segmentation=enable_segmentation,
            detection_confidence=detection_confidence,
            face_selection=face_selection
        )

        if len(masks) == 0:
            empty = torch.zeros_like(img_tensor)
            zero_mask = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)
            info = {
                "original_image": img_tensor.cpu().numpy(),
                "original_image_size": (img_tensor.shape[1], img_tensor.shape[2]),
                "crop_coords": (0, 0, 0, 0),
                "target_size": (processing_resolution, processing_resolution),
                "blend_mask": zero_mask.squeeze(0).numpy(),
                "original_crop_size": (0, 0)
            }
            return (empty, zero_mask, info)

        mask_np = masks[0].astype(np.float32)

        if mask_blur > 0:
            mask_np = cv2.GaussianBlur(mask_np, (mask_blur*2+1, mask_blur*2+1), 0)

        mask_proc = ForbiddenVisionMaskProcessor()

        face_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        cropped_face, sampler_mask, restore_info = mask_proc.process_and_crop(
            img_tensor,
            face_tensor,
            crop_padding,
            processing_resolution,
            mask_expansion,
            enable_pre_upscale=False,
            upscaler_model_name=None,
            upscaler_loader_callback=None,
            upscaler_run_callback=None
        )

        full_mask = face_tensor
        full_mask = torch.from_numpy(mask_np).unsqueeze(0)

        restore_info["original_image"] = img_tensor.cpu().numpy()
        restore_info["target_size"] = (processing_resolution, processing_resolution)

        return (cropped_face, full_mask, restore_info)
