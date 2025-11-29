import torch
import numpy as np
import comfy.model_management as model_management
from .face_detector import ForbiddenVisionFaceDetector
from .utils import check_for_interruption, ensure_model_directories, clean_model_name

class ForbiddenVisionFaceFixerMaskOnly:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for face mask detection."}),
                "face_selection": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "0=All faces, 1=1st face, 2=2nd face, etc."}),
                "enable_segmentation": ("BOOLEAN", {"default": True, "tooltip": "Use AI segmentation. If disabled, creates oval masks."}),
                "detection_confidence": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Face detection confidence threshold."}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("face_masks",)
    FUNCTION = "generate_face_masks"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        ensure_model_directories()
        self.face_detector = ForbiddenVisionFaceDetector()

    def generate_face_masks(self, image, face_selection, enable_segmentation, detection_confidence):
        try:
            check_for_interruption()
            
            if image is None:
                print("ERROR: No image input provided for mask generation.")
                return (torch.zeros((1, 512, 512), dtype=torch.float32, device=model_management.get_torch_device()),)

            np_masks = self.face_detector.detect_faces(
                image_tensor=image,
                enable_segmentation=enable_segmentation,
                detection_confidence=detection_confidence,
                face_selection=face_selection
            )

            if not np_masks:
                print("[Face Mask Generator] No faces detected. Returning empty mask.")
                h, w = image.shape[1], image.shape[2]
                device = image.device
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
                return (empty_mask,)

            print(f"[Face Mask Generator] Successfully detected {len(np_masks)} face mask(s).")

            face_masks = [torch.from_numpy(m).unsqueeze(0) for m in np_masks]
            
            h, w = image.shape[1], image.shape[2]
            device = image.device
            combined_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
            
            for mask_tensor in face_masks:
                mask_tensor = mask_tensor.to(device)
                combined_mask = torch.maximum(combined_mask, mask_tensor)

            mask_pixels = torch.sum(combined_mask > 0.5).item()
            print(f"[Face Mask Generator] Final combined mask: {mask_pixels} pixels")

            return (combined_mask,)

        except model_management.InterruptProcessingException:
            print("[Face Mask Generator] Processing interrupted by user.")
            raise
        except Exception as e:
            print(f"[Face Mask Generator] Error during mask generation: {e}")
            try:
                h, w = image.shape[1], image.shape[2] if image is not None else (512, 512)
                device = image.device if image is not None else model_management.get_torch_device()
                fallback_mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
                return (fallback_mask,)
            except:
                return (torch.zeros((1, 512, 512), dtype=torch.float32, device=model_management.get_torch_device()),)