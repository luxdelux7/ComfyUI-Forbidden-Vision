import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .utils import check_for_interruption
from .model_manager import ForbiddenVisionModelManager
from PIL import Image
import os
import time
from datetime import datetime

class ForbiddenVisionFaceDetector:
    
    def __init__(self):
        self.model_manager = ForbiddenVisionModelManager.get_instance()
        self.debug_dir = None
        self.debug_session_id = None
        self.debug_log_path = None
        self.full_image_for_processing = None
    
    
    def _create_fallback_mask(self, image_tensor, bbox=None):
        """Create a simple fallback mask when detection fails"""
        try:
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            
            if bbox:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((h, w), dtype=np.uint8)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
                return mask.astype(np.float32)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                axes = (w // 4, h // 3)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
                return mask.astype(np.float32)
                
        except Exception as e:
            self._debug_log(f"Error creating fallback mask: {e}", "ERROR")
            return np.zeros((512, 512), dtype=np.float32)
        
    def _debug_log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(log_entry)
        except:
            pass
        
        if level in ["ERROR", "CRITICAL"]:
            print(f"DEBUG {level}: {message}")

  


    def detect_faces(self, image_tensor, enable_segmentation=True, detection_confidence=0.6, face_selection=0):
        try:
            
            check_for_interruption()
            
            image_uint8 = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            self.full_image_for_processing = image_uint8
            original_h, original_w = image_uint8.shape[:2]
            
            
            detection_model = self.model_manager.load_face_detection_model()
            if not detection_model:
                self._debug_log("No face detection model available - creating fallback mask", "ERROR")
                fallback_mask = self._create_fallback_mask(image_tensor)
                return [fallback_mask] if face_selection == 0 or face_selection == 1 else []
            
            yolo_image, yolo_scale, yolo_offset = self.model_manager.resize_image_for_yolo(image_uint8)
            
            try:
                yolo_pil = Image.fromarray(yolo_image, mode='RGB')
                results = detection_model(yolo_pil, conf=detection_confidence, verbose=False)
            except Exception as detection_error:
                self._debug_log(f"YOLO detection failed: {detection_error}", "ERROR")
                fallback_mask = self._create_fallback_mask(image_tensor)
                return [fallback_mask] if face_selection == 0 or face_selection == 1 else []
            
            face_masks = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    original_bbox = self.model_manager.scale_bbox_back(box, yolo_scale, yolo_offset)
                    x1, y1, x2, y2 = original_bbox
                    
                    
                    crop_x1, crop_y1, crop_x2, crop_y2, scale_factor = self.model_manager.calculate_face_crop_region(
                        original_bbox, original_w, original_h
                    )
                    crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
                    
                    
                    face_crop = self.model_manager.extract_crop_with_padding(image_uint8, crop_coords)
                    
                    mask = None
                    if enable_segmentation:
                        try:
                            crop_mask = self.model_manager.segment_face(face_crop)
                            if crop_mask is not None:
                                mask = self._map_crop_mask_to_original(crop_mask, crop_coords, original_w, original_h)
                        except Exception as seg_error:
                            self._debug_log(f"Face {i}: segmentation failed: {seg_error}", "ERROR")

                    if mask is None:
                        mask = self.model_manager.create_oval_mask(original_bbox, original_h, original_w).astype(np.float32)
                        self._debug_log(f"Face {i}: using oval mask fallback")
                    
                    face_masks.append(mask)
               
            
            if not face_masks:
                self._debug_log("No faces detected, creating fallback mask", "WARNING")
                fallback_mask = self._create_fallback_mask(image_tensor)
                face_masks = [fallback_mask]
            
            if face_selection == 0:
                result_masks = face_masks
            elif face_masks and face_selection <= len(face_masks):
                result_masks = [face_masks[face_selection - 1]]
            elif face_masks:
                result_masks = [face_masks[0]]
            else:
                result_masks = []
            
            return result_masks
            
        except Exception as e:
            self._debug_log(f"Critical error in face detection: {e}", "ERROR")
            try:
                fallback_mask = self._create_fallback_mask(image_tensor)
                return [fallback_mask]
            except:
                return [np.zeros((512, 512), dtype=np.float32)]

    def _map_crop_mask_to_original(self, crop_mask, crop_coords, original_w, original_h):
        """Map crop mask back to original image coordinates"""
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_mask.shape != (crop_h, crop_w):
            resized_mask = cv2.resize(crop_mask.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = crop_mask
        
        full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        src_x1 = max(0, -crop_x1)
        src_y1 = max(0, -crop_y1)
        src_x2 = src_x1 + min(crop_w, original_w - max(0, crop_x1))
        src_y2 = src_y1 + min(crop_h, original_h - max(0, crop_y1))
        
        dst_x1 = max(0, crop_x1)
        dst_y1 = max(0, crop_y1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = resized_mask[src_y1:src_y2, src_x1:src_x2]
        
        return full_mask.astype(np.float32)