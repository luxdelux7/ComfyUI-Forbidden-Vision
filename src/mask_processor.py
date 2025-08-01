import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .utils import check_for_interruption

class ForbiddenVisionMaskProcessor:
    def __init__(self):
        pass

    def polish_mask(self, mask_np):
        try:
            if mask_np is None or np.sum(mask_np) == 0:
                return mask_np

            from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk

            mask_bool = (mask_np > 0.5)

            total_pixels = np.sum(mask_bool)
            speck_threshold = min(100, max(20, int(total_pixels * 0.0005)))
            cleaned_mask_bool = remove_small_objects(mask_bool, min_size=speck_threshold)
            
            hole_threshold = min(150, max(30, int(total_pixels * 0.001)))
            filled_mask_bool = remove_small_holes(cleaned_mask_bool, area_threshold=hole_threshold)
            
            
            rows = np.any(filled_mask_bool, axis=1)
            cols = np.any(filled_mask_bool, axis=0)
            if not np.any(rows) or not np.any(cols):
                return filled_mask_bool.astype(np.float32)

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            w, h = xmax - xmin, ymax - ymin

            closing_radius = int(min(w, h) * 0.02)
            closing_radius = np.clip(closing_radius, 2, 8)
            
            closed_mask = closing(filled_mask_bool, footprint=disk(closing_radius))

            opening_radius = 2
            
            final_mask_bool = opening(closed_mask, footprint=disk(opening_radius))

            return final_mask_bool.astype(np.float32)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Warning: Error during final mask polishing: {e}. Returning unpolished mask.")
            return mask_np
                
    def process_and_crop(self, image_tensor, mask_tensor, crop_padding, processing_resolution, mask_expansion, enable_pre_upscale=False, upscaler_model_name=None, upscaler_loader_callback=None, upscaler_run_callback=None):
        try:
            check_for_interruption()
            
            is_adaptive = isinstance(processing_resolution, tuple)
            if is_adaptive:
                target_wh = processing_resolution
            else:
                target_wh = (processing_resolution, processing_resolution)

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
                return self.create_empty_outputs(image_tensor, target_wh)

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
                return self.create_empty_outputs(image_tensor, target_wh)

            m_y1, m_y2 = final_coords[0].min(), final_coords[0].max()
            m_x1, m_x2 = final_coords[1].min(), final_coords[1].max()
            center_x, center_y = (m_x1 + m_x2) / 2, (m_y1 + m_y2) / 2
            
            if is_adaptive:
                mask_w, mask_h = m_x2 - m_x1, m_y2 - m_y1
                target_aspect_ratio = target_wh[0] / target_wh[1]
                
                padded_w = mask_w * crop_padding
                padded_h = mask_h * crop_padding
                
                current_aspect_ratio = padded_w / padded_h
                if current_aspect_ratio > target_aspect_ratio:
                    final_crop_w = padded_w
                    final_crop_h = padded_w / target_aspect_ratio
                else:
                    final_crop_h = padded_h
                    final_crop_w = padded_h * target_aspect_ratio

                if final_crop_w > w:
                    scale_factor = w / final_crop_w
                    final_crop_w *= scale_factor
                    final_crop_h *= scale_factor
                if final_crop_h > h:
                    scale_factor = h / final_crop_h
                    final_crop_w *= scale_factor
                    final_crop_h *= scale_factor

                crop_x1 = int(center_x - final_crop_w / 2)
                crop_y1 = int(center_y - final_crop_h / 2)
            
            else:
                width, height = m_x2 - m_x1, m_y2 - m_y1
                base_size = max(width, height)
                padded_size = int(base_size * crop_padding)
                final_crop_w = min(padded_size, w, h)
                final_crop_h = final_crop_w

                crop_x1 = int(center_x - final_crop_w / 2)
                crop_y1 = int(center_y - final_crop_h / 2)

            crop_x1 = max(0, min(crop_x1, w - int(final_crop_w)))
            crop_y1 = max(0, min(crop_y1, h - int(final_crop_h)))
            crop_x2 = crop_x1 + int(final_crop_w)
            crop_y2 = crop_y1 + int(final_crop_h)
            
            cropped_image = image_uint8[crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_sampler_mask = blend_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            
            should_upscale = (enable_pre_upscale and 
                            upscaler_model_name and 
                            upscaler_model_name != "None Found" and
                            upscaler_loader_callback and 
                            upscaler_run_callback and
                            max(cropped_image.shape) < max(target_wh) * 0.70)

            if should_upscale:
                if upscaler_loader_callback(upscaler_model_name):
                    upscaled_image = upscaler_run_callback(cropped_image)
                    cropped_image_resized = cv2.resize(upscaled_image, target_wh, interpolation=cv2.INTER_LANCZOS4)
                else:
                    print(f"Failed to load upscaler model {upscaler_model_name}, using standard resize")
                    cropped_image_resized = cv2.resize(cropped_image, target_wh, interpolation=cv2.INTER_LANCZOS4)
            else:
                cropped_image_resized = cv2.resize(cropped_image, target_wh, interpolation=cv2.INTER_LANCZOS4)
            
            sampler_mask_resized = cv2.resize(cropped_sampler_mask, target_wh, interpolation=cv2.INTER_LINEAR)

            cropped_face_tensor = torch.from_numpy((cropped_image_resized / 255.0).astype(np.float32)).unsqueeze(0)
            sampler_mask_tensor = torch.from_numpy(sampler_mask_resized).unsqueeze(0)

            restore_info = { 
                "original_image": image_tensor.cpu().numpy(), 
                "original_image_size": (h, w), 
                "crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), 
                "face_bbox": mask_bbox,
                "target_size": target_wh,
                "original_crop_size": (crop_x2 - crop_x1, crop_y2 - crop_y1),
                "blend_mask": blend_mask,
                "detection_angle": 0
            }

            return (cropped_face_tensor, sampler_mask_tensor, restore_info)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during mask processing and cropping: {e}")
            if isinstance(processing_resolution, int):
                target_wh = (processing_resolution, processing_resolution)
            else:
                target_wh = processing_resolution
            return self.create_empty_outputs(image_tensor, target_wh)

    def create_empty_outputs(self, image_tensor, target_size):
        if isinstance(target_size, int):
            target_wh = (target_size, target_size)
        else:
            target_wh = target_size

        empty_face = torch.zeros((1, target_wh[1], target_wh[0], 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, target_wh[1], target_wh[0]), dtype=torch.float32)
        
        empty_info = {
            "original_image": image_tensor.cpu().numpy() if image_tensor is not None else np.zeros((target_wh[1], target_wh[0], 3)),
            "original_image_size": (0, 0), "crop_coords": (0, 0, 0, 0),
            "face_bbox": (0, 0, 0, 0), "target_size": target_wh,
            "original_crop_size": (0, 0), 
            "blend_mask": np.zeros((target_wh[1], target_wh[0]), dtype=np.float32),
            "detection_angle": 0
        }
        return (empty_face, empty_mask, empty_info)