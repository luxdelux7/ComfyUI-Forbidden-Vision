import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .utils import check_for_interruption, safe_tensor_to_numpy

class ForbiddenVisionMaskProcessor:
    def __init__(self):
        pass

    def polish_mask(self, mask_np):
        """
        Performs a final, robust polishing of a generated mask using a two-pass strategy.
        This is designed to effectively remove noise and smooth jagged edges.
        """
        try:
            if mask_np is None or np.sum(mask_np) == 0:
                return mask_np

            from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk

            # Ensure we are working with a boolean mask for skimage functions
            mask_bool = (mask_np > 0.5)

            # 1. Initial cleanup of disconnected specks and pinholes
            total_pixels = np.sum(mask_bool)
            speck_threshold = min(100, max(20, int(total_pixels * 0.0005)))
            cleaned_mask_bool = remove_small_objects(mask_bool, min_size=speck_threshold)
            
            hole_threshold = min(150, max(30, int(total_pixels * 0.001)))
            filled_mask_bool = remove_small_holes(cleaned_mask_bool, area_threshold=hole_threshold)
            
            # 2. Two-Pass Polishing Strategy
            # This is more effective than a single operation.
            
            # Find the mask's bounding box to create dynamic structuring elements
            rows = np.any(filled_mask_bool, axis=1)
            cols = np.any(filled_mask_bool, axis=0)
            if not np.any(rows) or not np.any(cols):
                return filled_mask_bool.astype(np.float32) # Return early if mask is empty

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            w, h = xmax - xmin, ymax - ymin

            # --- Pass 1: Aggressive Closing to absorb fuzz and connect gaps ---
            # Use a larger, dynamic disk to make the shape solid.
            closing_radius = int(min(w, h) * 0.02) # 2% of the smallest dimension
            closing_radius = np.clip(closing_radius, 2, 8) # Clamp for safety (radius of 2-8 pixels)
            
            closed_mask = closing(filled_mask_bool, footprint=disk(closing_radius))

            # --- Pass 2: Gentle Opening to smooth the contour ---
            # Use a smaller, fixed disk to refine the edges without shrinking the main body.
            opening_radius = 2 # A fixed radius of 2 is effective for smoothing without being destructive
            
            final_mask_bool = opening(closed_mask, footprint=disk(opening_radius))

            return final_mask_bool.astype(np.float32)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Warning: Error during final mask polishing: {e}. Returning unpolished mask.")
            return mask_np
                
    def process_and_crop(self, image_tensor, mask_tensor, crop_padding, processing_resolution, mask_expansion, enable_pre_upscale=False, upscaler_model_name=None, upscaler_loader_callback=None, upscaler_run_callback=None):
        """
        Orchestrates mask processing: cropping and resizing for inpainting.
        """
        try:
            check_for_interruption()
            
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            if mask_tensor.device != torch.device('cpu'):
                mask_tensor = mask_tensor.cpu()

            image_np = image_tensor.squeeze().numpy()
            if image_np.max() <= 1.0:
                image_uint8 = (image_np * 255.0).astype(np.uint8)
            else:
                image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)

            mask_np = mask_tensor.squeeze().numpy()
            if mask_np.max() > 1.0:
                mask_float = mask_np / 255.0
            else:
                mask_float = mask_np
            h, w = image_uint8.shape[:2]
            
            if len(mask_float.shape) > 2:
                mask_float = mask_float.squeeze()

            initial_coords = np.where(mask_float > 0.5)
            if len(initial_coords[0]) == 0:
                print("Warning: Input mask is empty. Cannot process.")
                return self.create_empty_outputs(image_tensor, processing_resolution)

            y1, y2 = initial_coords[0].min(), initial_coords[0].max()
            x1, x2 = initial_coords[1].min(), initial_coords[1].max()
            mask_bbox = (x1, y1, x2, y2)
            
            blend_mask = mask_float

            if mask_expansion > 0:
                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_expansion*2+1, mask_expansion*2+1))
                blend_mask = cv2.dilate((blend_mask * 255).astype(np.uint8), expand_kernel, iterations=1)
                blend_mask = blend_mask.astype(np.float32) / 255.0
            
            final_coords = np.where(blend_mask > 0.5)
            if len(final_coords[0]) == 0:
                print("Warning: Mask became empty after processing. Cannot continue.")
                return self.create_empty_outputs(image_tensor, processing_resolution)

            m_y1, m_y2 = final_coords[0].min(), final_coords[0].max()
            m_x1, m_x2 = final_coords[1].min(), final_coords[1].max()

            center_x, center_y = (m_x1 + m_x2) / 2, (m_y1 + m_y2) / 2
            width, height = m_x2 - m_x1, m_y2 - m_y1

            base_size = max(width, height)
            padded_size = int(base_size * crop_padding)
            final_crop_size = min(padded_size, w, h)

            ideal_x1 = int(center_x - final_crop_size / 2)
            ideal_y1 = int(center_y - final_crop_size / 2)
            
            crop_x1 = max(0, min(ideal_x1, w - final_crop_size))
            crop_y1 = max(0, min(ideal_y1, h - final_crop_size))
            crop_x2 = crop_x1 + final_crop_size
            crop_y2 = crop_y1 + final_crop_size
            
            cropped_image = image_uint8[crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_sampler_mask = blend_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            should_upscale = (enable_pre_upscale and 
                            upscaler_model_name and 
                            upscaler_model_name != "None Found" and
                            upscaler_loader_callback and 
                            upscaler_run_callback and
                            final_crop_size < processing_resolution * 0.70)

            if should_upscale:
                if upscaler_loader_callback(upscaler_model_name):
                    upscaled_image = upscaler_run_callback(cropped_image)
                    cropped_image_resized = cv2.resize(upscaled_image, (processing_resolution, processing_resolution), interpolation=cv2.INTER_LANCZOS4)
                else:
                    print(f"Failed to load upscaler model {upscaler_model_name}, using standard resize")
                    cropped_image_resized = cv2.resize(cropped_image, (processing_resolution, processing_resolution), interpolation=cv2.INTER_LANCZOS4)
            else:
                cropped_image_resized = cv2.resize(cropped_image, (processing_resolution, processing_resolution), interpolation=cv2.INTER_LANCZOS4)
            sampler_mask_resized = cv2.resize(cropped_sampler_mask, (processing_resolution, processing_resolution), interpolation=cv2.INTER_LINEAR)

            cropped_face_tensor = torch.from_numpy((cropped_image_resized / 255.0).astype(np.float32)).unsqueeze(0)
            sampler_mask_tensor = torch.from_numpy(sampler_mask_resized).unsqueeze(0)

            restore_info = { 
                "original_image": image_tensor.cpu().numpy(), 
                "original_image_size": (h, w), 
                "crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), 
                "face_bbox": mask_bbox,
                "target_size": processing_resolution, 
                "original_crop_size": (final_crop_size, final_crop_size), 
                "blend_mask": blend_mask,
                "detection_angle": 0
            }

            return (cropped_face_tensor, sampler_mask_tensor, restore_info)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during mask processing and cropping: {e}")
            import traceback
            traceback.print_exc()
            return self.create_empty_outputs(image_tensor, processing_resolution)

    def create_empty_outputs(self, image_tensor, target_size):
        empty_face = torch.zeros((1, target_size, target_size, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, target_size, target_size), dtype=torch.float32)
        empty_info = {
            "original_image": image_tensor.cpu().numpy() if image_tensor is not None else np.zeros((target_size, target_size, 3)),
            "original_image_size": (0, 0), "crop_coords": (0, 0, 0, 0),
            "face_bbox": (0, 0, 0, 0), "target_size": target_size,
            "original_crop_size": (0, 0), 
            "blend_mask": np.zeros((target_size, target_size), dtype=np.float32),
            "detection_angle": 0
        }
        return (empty_face, empty_mask, empty_info)