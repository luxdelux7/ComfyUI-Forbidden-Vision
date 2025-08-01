import torch
import numpy as np
import cv2
import math
from ensemble_boxes import weighted_boxes_fusion
import comfy.model_management as model_management
from .utils import check_for_interruption, find_model_path, clean_model_name
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from PIL import Image

class ForbiddenVisionFaceDetector:
    def __init__(self):
        self.sam_model, self.sam_predictor, self.sam_model_name = None, None, None
        self.bbox_model, self.bbox_model_name = None, None
        self.bbox_model_B, self.bbox_model_B_name = None, None
        self.yoloseg_model, self.yoloseg_model_name = None, None
    
    def _can_share_model(self, clean_model_name, model_type_needed):
        if model_type_needed == 'yoloseg':
            return None
        
        if model_type_needed == 'bbox':
            if (self.bbox_model_B is not None and self.bbox_model_B_name == clean_model_name):
                return 'bbox_B'
        elif model_type_needed == 'bbox_B':
            if (self.bbox_model is not None and self.bbox_model_name == clean_model_name):
                return 'bbox'
        
        return None

    def _share_model_instance(self, source_type, target_type, clean_model_name):

        if source_type == 'bbox' and target_type == 'yoloseg':
            self.yoloseg_model = self.bbox_model
            self.yoloseg_model_name = clean_model_name
            return True
        elif source_type == 'bbox_B' and target_type == 'yoloseg':
            self.yoloseg_model = self.bbox_model_B
            self.yoloseg_model_name = clean_model_name
            return True
        elif source_type == 'yoloseg' and target_type == 'bbox':
            self.bbox_model = self.yoloseg_model
            self.bbox_model_name = clean_model_name
            return True
        elif source_type == 'yoloseg' and target_type == 'bbox_B':
            self.bbox_model_B = self.yoloseg_model
            self.bbox_model_B_name = clean_model_name
            return True
        elif source_type == 'bbox' and target_type == 'bbox_B':
            self.bbox_model_B = self.bbox_model
            self.bbox_model_B_name = clean_model_name
            return True
        
        return False
    
    def load_bbox_detector(self, model_name):
        try:
            clean_model_name_val = clean_model_name(model_name)
            if self.bbox_model is not None and self.bbox_model_name == clean_model_name_val: return True
            
            share_source = self._can_share_model(clean_model_name_val, 'bbox')
            if share_source:
                return self._share_model_instance(share_source, 'bbox', clean_model_name_val)
            
            model_path = find_model_path(clean_model_name_val, 'yolo')
            if model_path is None: print(f"Error: BBOX (YOLO) model '{clean_model_name_val}' not found."); return False
            from ultralytics import YOLO
            device = model_management.get_torch_device()
            if self.bbox_model is not None: del self.bbox_model
            self.bbox_model = YOLO(model_path); self.bbox_model.to(device)
            self.bbox_model_name = clean_model_name_val
            return True
        except Exception as e: print(f"Error loading YOLO '{clean_model_name_val}': {e}"); self.bbox_model = None; self.bbox_model_name = None; return False
    
    def load_bbox_detector_B(self, model_name):
        try:
            clean_model_name_val = clean_model_name(model_name)
            if self.bbox_model_B is not None and self.bbox_model_B_name == clean_model_name_val: return True
            
            share_source = self._can_share_model(clean_model_name_val, 'bbox_B')
            if share_source:
                return self._share_model_instance(share_source, 'bbox_B', clean_model_name_val)
            
            model_path = find_model_path(clean_model_name_val, 'yolo')
            if model_path is None: print(f"Error: BBOX (YOLO) model '{clean_model_name_val}' not found."); return False
            from ultralytics import YOLO
            device = model_management.get_torch_device()
            if self.bbox_model_B is not None: del self.bbox_model_B
            self.bbox_model_B = YOLO(model_path); self.bbox_model_B.to(device)
            self.bbox_model_B_name = clean_model_name_val
            return True
        except Exception as e: print(f"Error loading YOLO '{clean_model_name_val}': {e}"); self.bbox_model_B = None; self.bbox_model_B_name = None; return False
    
    def load_yoloseg_model(self, model_name):
        try:
            clean_model_name_val = clean_model_name(model_name)
            if self.yoloseg_model is not None and self.yoloseg_model_name == clean_model_name_val: return True
            
            share_source = self._can_share_model(clean_model_name_val, 'yoloseg')
            if share_source:
                return self._share_model_instance(share_source, 'yoloseg', clean_model_name_val)
            
            model_path = find_model_path(clean_model_name_val, 'yolo')
            if model_path is None: print(f"Error: YOLO-seg model '{clean_model_name_val}' not found."); return False
            from ultralytics import YOLO
            device = model_management.get_torch_device()
            if self.yoloseg_model is not None: del self.yoloseg_model
            self.yoloseg_model = YOLO(model_path); self.yoloseg_model.to(device)
            self.yoloseg_model_name = clean_model_name_val
            return True
        except Exception as e: print(f"Error loading YOLO-seg '{clean_model_name_val}': {e}"); self.yoloseg_model = None; self.yoloseg_model_name = None; return False
        
    def load_sam_model(self, model_name):
        try:
            clean_model_name_val = clean_model_name(model_name)
            if self.sam_model is not None and self.sam_model_name == clean_model_name_val: return True
            model_path = find_model_path(clean_model_name_val, 'sam')
            if model_path is None: print(f"Error: SAM model '{clean_model_name_val}' not found."); return False
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "default"
            if "vit_b" in clean_model_name_val: model_type = "vit_b"
            elif "vit_l" in clean_model_name_val: model_type = "vit_l"
            elif "vit_h" in clean_model_name_val: model_type = "vit_h"
            device = model_management.get_torch_device()
            self.sam_model = sam_model_registry[model_type](checkpoint=model_path); self.sam_model.to(device)
            self.sam_predictor = SamPredictor(self.sam_model)
            self.sam_model_name = clean_model_name_val
            return True
        except Exception as e: print(f"Error loading SAM model: {e}"); self.sam_model=None; self.sam_predictor=None; self.sam_model_name=None; return False

    def calculate_solidity_score(self, mask_np):
        try:
            if mask_np.sum() == 0: return 0.0
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.0
            mask_pixel_area = np.sum(mask_uint8)
            if mask_pixel_area == 0: return 0.0
            all_points = np.concatenate(contours, axis=0)
            hull = cv2.convexHull(all_points)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: return 0.0
            return mask_pixel_area / hull_area
        except Exception: return 0.5

    def calculate_face_coverage_ratio(self, mask_np, tight_bbox):
        try:
            if mask_np.sum() == 0: return 0.0
            x1, y1, x2, y2 = tight_bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0: return 0.0
            
            mask_pixels_in_bbox = np.sum(mask_np[y1:y2, x1:x2] > 0.5)
            coverage_ratio = mask_pixels_in_bbox / bbox_area
            
            if coverage_ratio < 0.4:
                return coverage_ratio
            elif coverage_ratio > 0.85:
                return 0.85 - (coverage_ratio - 0.85) * 2
            else:
                return 0.9 + 0.1 * (1.0 - abs(coverage_ratio - 0.7) / 0.3)
        except Exception:
            return 0.5

    def make_crop_region(self, bbox, w, h, crop_factor=2.0):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        ew, eh = int(bw * (crop_factor-1.0)/2.0), int(bh * (crop_factor-1.0)/2.0)
        return [max(0, x1 - ew), max(0, y1 - eh), min(w, x2 + ew), min(h, y2 + eh)]
    
   
    def select_best_mask(self, all_masks_data, tight_bbox):

        if not all_masks_data:
            return None

        x1, y1, x2, y2 = tight_bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        scored_candidates = []
        for data in all_masks_data:
            mask = data['original_mask']
            h, w = mask.shape
            
            total_mask_area = np.sum(mask > 0.5)
            if total_mask_area == 0: 
                continue

            inside_bbox_area = np.sum(mask[y1:y2, x1:x2] > 0.5)
            
            outside_bbox_area = 0
            if y1 > 0: outside_bbox_area += np.sum(mask[0:y1, :] > 0.5)
            if y2 < h: outside_bbox_area += np.sum(mask[y2:h, :] > 0.5)
            if x1 > 0: outside_bbox_area += np.sum(mask[max(0,y1):min(h,y2), 0:x1] > 0.5)
            if x2 < w: outside_bbox_area += np.sum(mask[max(0,y1):min(h,y2), x2:w] > 0.5)
            
            outside_ratio = outside_bbox_area / total_mask_area if total_mask_area > 0 else 0
            
            if outside_ratio <= 0.02: containment_score = 1.0
            elif outside_ratio <= 0.05: containment_score = 0.8
            elif outside_ratio <= 0.10: containment_score = 0.6
            elif outside_ratio <= 0.15: containment_score = 0.4
            else: containment_score = 0.2
            
            coverage_score = self.calculate_face_coverage_ratio(mask, tight_bbox)
            mask_uint8 = (mask > 0.5).astype(np.uint8)
            
            cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_for_analysis = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, cleanup_kernel)
            cleaned_for_analysis = cv2.morphologyEx(cleaned_for_analysis, cv2.MORPH_CLOSE, cleanup_kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_for_analysis, 4, cv2.CV_32S)
            
            edge_noise = 0
            edge_thickness = 2
            if y1 + edge_thickness < y2: edge_noise += np.sum(mask[y1:y1+edge_thickness, x1:x2] > 0.5)
            if y2 - edge_thickness > y1: edge_noise += np.sum(mask[y2-edge_thickness:y2, x1:x2] > 0.5)
            if x1 + edge_thickness < x2: edge_noise += np.sum(mask[y1:y2, x1:x1+edge_thickness] > 0.5)
            if x2 - edge_thickness > x1: edge_noise += np.sum(mask[y1:y2, x2-edge_thickness:x2] > 0.5)
            
            edge_noise_ratio = edge_noise / total_mask_area if total_mask_area > 0 else 0
            
            if num_labels <= 5 and edge_noise_ratio < 0.05: cleanliness_score = 1.0
            elif num_labels <= 10 and edge_noise_ratio < 0.10: cleanliness_score = 0.8
            elif num_labels <= 20 and edge_noise_ratio < 0.15: cleanliness_score = 0.6
            elif num_labels <= 40 and edge_noise_ratio < 0.25: cleanliness_score = 0.4
            else: cleanliness_score = 0.2
            
            if outside_bbox_area == 0 and edge_noise_ratio == 0: cleanliness_score = min(1.0, cleanliness_score + 0.2)
            if num_labels > 60: cleanliness_score *= 0.7
            
            shape_score = self.calculate_comprehensive_shape_score(mask)
            
            final_score = (containment_score * 0.50 + coverage_score * 0.10 + cleanliness_score * 0.30 + shape_score * 0.10)
            
            if outside_bbox_area == 0: final_score += 0.04
            if edge_noise_ratio < 0.01: final_score += 0.02
            final_score += data.get('sam_score', 0) * 0.01
            
            data['final_score'] = final_score
            scored_candidates.append(data)
            

        if not scored_candidates:

            return None

        sorted_candidates = sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)
        
        core_x1 = x1 + int(bbox_width * 0.25)
        core_y1 = y1 + int(bbox_height * 0.25)
        core_x2 = x2 - int(bbox_width * 0.25)
        core_y2 = y2 - int(bbox_height * 0.25)
        core_area = (core_x2 - core_x1) * (core_y2 - core_y1)
        CORE_FILL_THRESHOLD = 0.3
        
        best_mask_data = None
        for candidate in sorted_candidates:
            if core_area > 0:
                core_fill = np.sum(candidate['original_mask'][core_y1:core_y2, core_x1:core_x2] > 0.5)
                core_fill_ratio = core_fill / core_area
            else:
                core_fill_ratio = 0.0
       
            if core_fill_ratio >= CORE_FILL_THRESHOLD:
                best_mask_data = candidate
                break

        if best_mask_data is None:
            best_mask_data = sorted_candidates[0]
            
        
        best_base = best_mask_data
        best_base['complementary_masks'] = [
            d for d in all_masks_data 
            if d['index'] != best_base['index']
        ]
        
        return best_base
    
    def generate_sam_masks(self, bbox):
        all_masks, all_scores = [], []
        
        x1, y1, x2, y2 = bbox
        cx, cy, qh = (x1+x2)/2, (y1+y2)/2, (y2-y1)*0.25
        points = np.array([[cx, cy - qh], [cx, cy], [cx, cy + qh]])
        labels = np.array([1, 1, 1])
        masks_points, scores_points, _ = self.sam_predictor.predict(point_coords=points, point_labels=labels, box=np.array(bbox), multimask_output=True)
        if masks_points is not None: all_masks.extend(masks_points); all_scores.extend(scores_points)

        masks_box, scores_box, _ = self.sam_predictor.predict(box=np.array(bbox), multimask_output=True)
        if masks_box is not None: all_masks.extend(masks_box); all_scores.extend(scores_box)
        
        return all_masks, all_scores
    
    def _cleanup_yoloseg_mask(self, mask_np):

        try:
            if mask_np.sum() == 0:
                return mask_np
            
            mask_bool = mask_np > 0
            mask_area = np.sum(mask_bool)
            h, w = mask_np.shape
            
            if mask_area < 50:
                return np.zeros_like(mask_np, dtype=np.uint8)
            
            image_diagonal = np.sqrt(h * h + w * w)
            
            area_based_threshold = max(15, int(mask_area * 0.01))
            resolution_based_threshold = max(25, int((h * w) * 0.0005))
            min_component_size = max(area_based_threshold, resolution_based_threshold)

            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), 8, cv2.CV_32S)
            
            if num_labels <= 1:
                return np.zeros_like(mask_np, dtype=np.uint8)
            
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_idx = np.argmax(component_areas) + 1
            largest_area = component_areas[largest_component_idx - 1]

            
            cleaned_mask = np.zeros_like(mask_bool, dtype=np.uint8)
            cleaned_mask[labels == largest_component_idx] = 1
            
            for i in range(1, num_labels):
                if i == largest_component_idx:
                    continue
                    
                component_area = stats[i, cv2.CC_STAT_AREA]
                
                if (component_area >= min_component_size and 
                    component_area >= largest_area * 0.05):
                    cleaned_mask[labels == i] = 1
            
            cleaned_mask_bool = cleaned_mask > 0
            hole_threshold = max(20, int(np.sum(cleaned_mask_bool) * 0.02))
            final_mask_bool = remove_small_holes(cleaned_mask_bool, area_threshold=hole_threshold)
            
            kernel_size = max(3, min(5, int(min(h, w) * 0.005)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            final_mask_uint8 = final_mask_bool.astype(np.uint8)
            smoothed_mask = cv2.morphologyEx(final_mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            removed_pixels = mask_area - np.sum(smoothed_mask)
            
            return smoothed_mask
            
        except Exception as e:
            print(f"Error during YOLO-seg mask cleanup: {e}")
            return (mask_np > 0).astype(np.uint8)
    
    def _select_best_yoloseg_mask(self, cleaned_masks, tight_bbox_in_crop):

        try:
            if not cleaned_masks:
                return None
            
            if len(cleaned_masks) == 1:
                return cleaned_masks[0]
            
            x1, y1, x2, y2 = tight_bbox_in_crop
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            edge_buffer = 3
            
            scored_masks = []
            
            for mask_data in cleaned_masks:
                mask = mask_data['mask']
                mask_area = np.sum(mask > 0)
                
                if mask_area == 0:
                    continue
                
                edge_violations = 0
                total_boundary_length = 0
                
                top_boundary_pixels = np.sum(mask[y1:y1+edge_buffer, x1:x2] > 0)
                top_boundary_length = x2 - x1
                total_boundary_length += top_boundary_length
                edge_violations += top_boundary_pixels
                
                bottom_boundary_pixels = np.sum(mask[max(y1, y2-edge_buffer):y2, x1:x2] > 0)
                bottom_boundary_length = x2 - x1
                total_boundary_length += bottom_boundary_length
                edge_violations += bottom_boundary_pixels
                
                left_boundary_pixels = np.sum(mask[y1:y2, x1:x1+edge_buffer] > 0)
                left_boundary_length = y2 - y1
                total_boundary_length += left_boundary_length
                edge_violations += left_boundary_pixels
                
                right_boundary_pixels = np.sum(mask[y1:y2, max(x1, x2-edge_buffer):x2] > 0)
                right_boundary_length = y2 - y1
                total_boundary_length += right_boundary_length
                edge_violations += right_boundary_pixels
                
                boundary_violation_ratio = edge_violations / total_boundary_length if total_boundary_length > 0 else 0
                
                mask_coords = np.argwhere(mask > 0)
                if len(mask_coords) > 0:
                    mask_y_coords, mask_x_coords = mask_coords[:, 0], mask_coords[:, 1]
                    
                    min_dist_to_top = np.min(mask_y_coords - y1) if len(mask_y_coords) > 0 else bbox_height
                    min_dist_to_bottom = np.min(y2 - mask_y_coords) if len(mask_y_coords) > 0 else bbox_height
                    min_dist_to_left = np.min(mask_x_coords - x1) if len(mask_x_coords) > 0 else bbox_width
                    min_dist_to_right = np.min(x2 - mask_x_coords) if len(mask_x_coords) > 0 else bbox_width
                    
                    boundary_proximity_penalty = 0
                    edge_threshold = 5
                    
                    if min_dist_to_top < edge_threshold:
                        boundary_proximity_penalty += (edge_threshold - min_dist_to_top) / edge_threshold * 0.25
                    if min_dist_to_bottom < edge_threshold:
                        boundary_proximity_penalty += (edge_threshold - min_dist_to_bottom) / edge_threshold * 0.25
                    if min_dist_to_left < edge_threshold:
                        boundary_proximity_penalty += (edge_threshold - min_dist_to_left) / edge_threshold * 0.25
                    if min_dist_to_right < edge_threshold:
                        boundary_proximity_penalty += (edge_threshold - min_dist_to_right) / edge_threshold * 0.25
                    
                    boundary_proximity_penalty = min(1.0, boundary_proximity_penalty)
                else:
                    boundary_proximity_penalty = 1.0
                
                coverage_ratio = mask_area / bbox_area
                
                if 0.35 <= coverage_ratio <= 0.60:
                    coverage_score = 1.0
                elif 0.25 <= coverage_ratio < 0.35:
                    coverage_score = 0.7 + (coverage_ratio - 0.25) * 3
                elif 0.60 < coverage_ratio <= 0.75:
                    coverage_score = 1.0 - (coverage_ratio - 0.60) * 2
                else:
                    coverage_score = max(0.1, 1.0 - abs(coverage_ratio - 0.45) * 2)
                
                compactness_score = self.calculate_solidity_score(mask)
                
                acceptable_violation_threshold = 0.08
                penalty_steepness = 12
                
                sigmoid_input = penalty_steepness * (boundary_violation_ratio - acceptable_violation_threshold)
                boundary_score = 1.0 / (1.0 + math.exp(sigmoid_input))
                
                boundary_score = max(0.05, boundary_score - boundary_proximity_penalty * 0.5)
                
                final_score = (
                    boundary_score * 0.70 +
                    coverage_score * 0.20 +
                    compactness_score * 0.10
                )
                
                scored_masks.append({
                    'mask_data': mask_data,
                    'score': final_score,
                    'boundary_violation_ratio': boundary_violation_ratio,
                    'boundary_proximity_penalty': boundary_proximity_penalty,
                    'boundary_score': boundary_score,
                    'coverage_ratio': coverage_ratio,
                    'coverage_score': coverage_score,
                    'edge_details': {
                        'top': top_boundary_pixels,
                        'bottom': bottom_boundary_pixels, 
                        'left': left_boundary_pixels,
                        'right': right_boundary_pixels,
                        'min_distances': [min_dist_to_top, min_dist_to_bottom, min_dist_to_left, min_dist_to_right]
                    }
                }) 
            
            if not scored_masks:
                return None
            
            best_mask = max(scored_masks, key=lambda x: x['score'])
            
            return best_mask['mask_data']
            
        except Exception as e:
            print(f"Error in YOLO-seg mask selection: {e}")
            return cleaned_masks[0] if cleaned_masks else None
    
    def _process_yoloseg_results(self, results, face_data, crop_w, crop_h):
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            mask_count = len(results[0].masks.data) if results and results[0].masks else 0
            return None
        
        mask_count = len(results[0].masks.data)
        
        if mask_count == 1:
            mask_tensor = results[0].masks.data[0].cpu()
            resized_mask_float = cv2.resize(
                mask_tensor.numpy().astype(np.float32),
                (crop_w, crop_h),
                interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (resized_mask_float > 0.4).astype(np.uint8)
            cleaned_mask = self._cleanup_yoloseg_mask(binary_mask)
            return cleaned_mask
        else:
            cleaned_masks = []
            
            for i in range(mask_count):
                mask_tensor = results[0].masks.data[i].cpu()
                resized_mask_float = cv2.resize(
                    mask_tensor.numpy().astype(np.float32),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR
                )
                
                binary_mask = (resized_mask_float > 0.4).astype(np.uint8)
                cleaned_mask = self._cleanup_yoloseg_mask(binary_mask)
                
                if np.sum(cleaned_mask) > 50:
                    cleaned_masks.append({
                        'index': i,
                        'mask': cleaned_mask,
                        'original_index': i
                    })
            
            if not cleaned_masks:
                return None
            
            if len(cleaned_masks) == 1:
                return cleaned_masks[0]['mask']
            
            bbox = face_data['bbox']
            cx1, cy1, cx2, cy2 = self.make_crop_region(bbox, self.full_image_for_sam.shape[1], self.full_image_for_sam.shape[0])
            tight_bbox_in_crop = [bbox[0]-cx1, bbox[1]-cy1, bbox[2]-cx1, bbox[3]-cy1]
            
            best_mask_data = self._select_best_yoloseg_mask(cleaned_masks, tight_bbox_in_crop)
            
            if not best_mask_data:
                return cleaned_masks[0]['mask']
            
            selected_mask = best_mask_data['mask']
            return selected_mask
        
    
    def _graft_satellite_masks(self, best_mask_data, all_masks_data, crop_bbox):
        try:
            check_for_interruption()
            
            anchor_mask = (best_mask_data['original_mask'] > 0.5)
            if np.sum(anchor_mask) == 0:
                return best_mask_data.get('mask', np.zeros_like(anchor_mask, dtype=np.uint8))
            
            vetted_candidates = []
   
            for c in all_masks_data:
                if c['index'] == best_mask_data['index']:
                    continue
                
                overall_solidity = self.calculate_solidity_score(c['original_mask'])
                if overall_solidity > 0.70:
                    vetted_candidates.append(c)
                    

            if not vetted_candidates:
                return anchor_mask.astype(np.uint8)

            anchor_inv = (anchor_mask == 0).astype(np.uint8)
            dist_map   = cv2.distanceTransform(anchor_inv, cv2.DIST_L2, 5)

            anchor_area = np.sum(anchor_mask)
            equivalent_diameter = 2 * np.sqrt(anchor_area / np.pi)
            
            min_area = 50 + (anchor_area * 0.0025)
            min_dist_threshold = 3.0 + (equivalent_diameter * 0.015)
            max_dist_threshold = equivalent_diameter * 0.30
            max_area = anchor_area * 0.50


            potential_grafts = []

            for candidate in vetted_candidates:
                check_for_interruption()

                candidate_mask = (candidate['original_mask'] > 0.5)
                new_parts_mask = candidate_mask & (~anchor_mask)
                
                if not np.any(new_parts_mask): continue

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    new_parts_mask.astype(np.uint8), 8, cv2.CV_32S
                )

                if num_labels <= 1: continue

                for i in range(1, num_labels):
                    component_mask = (labels == i)
                    component_area = stats[i, cv2.CC_STAT_AREA]

                    if not (min_area < component_area < max_area):
                        continue

                    min_dist = np.min(dist_map[component_mask])
                    
                    if min_dist >= max_dist_threshold:
                        continue
                    
                    proximity_score = 1.0 - (min_dist / max_dist_threshold)
                    solidity_score = self.calculate_solidity_score(component_mask)
                    final_graft_score = (proximity_score * 0.4) + (solidity_score * 0.6)
                    
                    if final_graft_score > 0.55:
                        potential_grafts.append({
                            'score': final_graft_score,
                            'mask': component_mask,
                            'dist': min_dist,
                            'area': component_area,
                            'solidity': solidity_score
                        })
                        
            if not potential_grafts:
                return anchor_mask.astype(np.uint8)

            sorted_grafts = sorted(potential_grafts, key=lambda x: x['score'], reverse=True)
            
            grafted_mask = anchor_mask.copy()
            for graft in sorted_grafts:
                if np.any(grafted_mask & graft['mask']):
                    continue

                grafted_mask = np.logical_or(grafted_mask, graft['mask'])

            return grafted_mask.astype(np.uint8)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"ERROR during satellite grafting: {e}")
            return (best_mask_data.get('mask', np.zeros((1,1))) > 0.5).astype(np.uint8)
    
    def calculate_comprehensive_shape_score(self, mask_np):
        try:
            if mask_np.sum() == 0: return 0.0
            
            coords = np.argwhere(mask_np > 0.5)
            if coords.shape[0] < 100: return 0.0
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            mask_bbox_area = (max_x - min_x) * (max_y - min_y)
            mask_pixel_area = coords.shape[0]
            compactness_score = mask_pixel_area / mask_bbox_area if mask_bbox_area > 0 else 0

            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.0
            
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            if perimeter == 0: return 0.0
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.4:
                circularity_score = max(0.0, circularity / 0.4)
            elif circularity > 0.9:
                circularity_score = max(0.0, 1.0 - (circularity - 0.9) * 2)
            else:
                circularity_score = 0.8 + 0.2 * (1.0 - abs(circularity - 0.7) / 0.3)
                
            final_score = (compactness_score * 0.7) + (circularity_score * 0.3)
            return np.clip(final_score, 0.0, 1.0)
                
        except Exception:
            return 0.5
    def analyze_mask_geometry(self, mask_np):
        try:
            if mask_np.sum() == 0:
                return None
                
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            main_contour = max(contours, key=cv2.contourArea)
            
            moments = cv2.moments(main_contour)
            if moments["m00"] == 0:
                return None
                
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            
            coords = np.argwhere(mask_np > 0.5)
            if coords.shape[0] == 0:
                return None
                
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            mask_height = max_y - min_y
            mask_width = max_x - min_x
            
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            mask_area = np.sum(mask_np > 0.5)
            
            ellipse = cv2.fitEllipse(main_contour) if len(main_contour) >= 5 else None
            
            return {
                'centroid': (centroid_x, centroid_y),
                'bbox': (min_x, min_y, max_x, max_y),
                'width': mask_width,
                'height': mask_height,
                'area': mask_area,
                'hull_area': hull_area,
                'main_contour': main_contour,
                'ellipse': ellipse,
                'hull': hull
            }
            
        except Exception as e:
            print(f"Error analyzing mask geometry: {e}")
            return None
    
    def _recompose_mask_with_detail_preservation(self, processed_mask, original_mask, min_hole_area_threshold, noise_threshold):
        try:
            
            final_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not final_contours:
                return processed_mask

            solid_silhouette = np.zeros_like(processed_mask)
            final_main_contour = max(final_contours, key=cv2.contourArea)
            cv2.drawContours(solid_silhouette, [final_main_contour], -1, color=1, thickness=cv2.FILLED)

            filled_details = cv2.subtract(solid_silhouette, original_mask)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_details, 8, cv2.CV_32S)
            
            major_fills_mask = np.zeros_like(filled_details)
            
            if num_labels <= 1:
                
                return processed_mask
            
            h, w = processed_mask.shape
            mask_area = np.sum(original_mask > 0)
            mask_center_x, mask_center_y = w // 2, h // 2
            
            
            for i in range(1, num_labels):
                component_area = stats[i, cv2.CC_STAT_AREA]
                
                if component_area < min_hole_area_threshold:
                    continue

                component_mask = (labels == i).astype(np.uint8)
                solidity = self.calculate_solidity_score(component_mask)
                
                cx, cy = centroids[i]
                distance_from_center = np.sqrt((cx - mask_center_x)**2 + (cy - mask_center_y)**2)
                max_distance = min(w, h) * 0.5
                centrality_score = max(0, 1.0 - (distance_from_center / max_distance))
                
                relative_size = component_area / mask_area if mask_area > 0 else 0
                
                edge_distance = min(cx, cy, w - cx, h - cy)
                edge_proximity = 1.0 - min(edge_distance / (min(w, h) * 0.1), 1.0)
                
                is_large_facial_feature = relative_size > 0.02 and centrality_score > 0.6
                is_medium_central_feature = relative_size > 0.005 and centrality_score > 0.7 and solidity > 0.3
                is_edge_intrusion = edge_proximity > 0.5 and relative_size < 0.01 and solidity > 0.7
                
                should_keep_filled = is_large_facial_feature or is_medium_central_feature
                should_restore_detail = is_edge_intrusion and not should_keep_filled
                
                if should_keep_filled:
                    major_fills_mask[labels == i] = 1
                elif should_restore_detail:
                    pass
                else:
                    major_fills_mask[labels == i] = 1

            recomposed_mask = cv2.bitwise_or(original_mask, major_fills_mask)

            mask_bool = recomposed_mask > 0
            cleaned_mask_bool = remove_small_objects(mask_bool, min_size=noise_threshold)
            cleaned_mask_bool = remove_small_holes(cleaned_mask_bool, area_threshold=noise_threshold)
            final_cleaned_mask = cleaned_mask_bool.astype(np.uint8)
            return final_cleaned_mask

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during final mask recomposition: {e}")
            return processed_mask
    
    def generate_sam_mask_for_face(self, face_data, face_index, sam_threshold, attempt_face_completion=False):
        try:
            check_for_interruption()

            bbox = face_data['bbox']
            full_image = self.full_image_for_sam
            fh, fw = full_image.shape[:2]
            cx1, cy1, cx2, cy2 = self.make_crop_region(bbox, fw, fh)
            crop_img = full_image[cy1:cy2, cx1:cx2]
            self.sam_predictor.set_image(crop_img, image_format='RGB')
            crop_bbox = [bbox[0]-cx1, bbox[1]-cy1, bbox[2]-cx1, bbox[3]-cy1]
            
            all_masks, all_scores = self.generate_sam_masks(crop_bbox)
            if not all_masks: 
                self.sam_predictor.set_image(full_image, image_format='RGB')
                return []

            all_masks_data = []
            for i, (m, s) in enumerate(zip(all_masks, all_scores)):
                if np.sum(m) == 0: continue
                all_masks_data.append({
                    'index': i,
                    'mask': (m > sam_threshold).astype(np.uint8),
                    'original_mask': (m > sam_threshold).astype(np.uint8),
                    'sam_score': s
                })

            best_base_data = self.select_best_mask(all_masks_data, crop_bbox)

            if not best_base_data:
                self.sam_predictor.set_image(full_image, image_format='RGB')
                return []

            if attempt_face_completion:
                grafted_mask = self._graft_satellite_masks(best_base_data, all_masks_data, crop_bbox)
            else:
                grafted_mask = (best_base_data['original_mask'] > 0.5).astype(np.uint8)

            UncleanedOriginalMask = grafted_mask
            BaseSilhouetteMask = UncleanedOriginalMask.copy()
            
            
            mask_geometry = self.analyze_mask_geometry(BaseSilhouetteMask.astype(np.float32))
            if mask_geometry:
                min_kernel_size = max(3, int(min(mask_geometry['width'], mask_geometry['height']) * 0.05))
                stencil_kernel_size = max(min_kernel_size, int(min(mask_geometry['width'], mask_geometry['height']) * 0.20))
                bbox_area = mask_geometry['width'] * mask_geometry['height']
                dynamic_noise_threshold = int(bbox_area * 0.0005) 
                min_noise_threshold = max(3, int(bbox_area * 0.0001))
                max_noise_threshold = max(50, int(bbox_area * 0.002))
                dynamic_noise_threshold = np.clip(dynamic_noise_threshold, min_noise_threshold, max_noise_threshold).item()
            else:
                fallback_size = 100
                min_kernel_size = max(3, int(fallback_size * 0.05))
                stencil_kernel_size = max(min_kernel_size, int(fallback_size * 0.20))
                dynamic_noise_threshold = max(3, int(fallback_size * 0.05))

            mask_bool = UncleanedOriginalMask > 0
            mask_area = np.sum(mask_bool)
            preliminary_threshold = max(10, int(mask_area * 0.001))
            preliminary_clean = remove_small_objects(mask_bool, min_size=preliminary_threshold)
            preliminary_clean = remove_small_holes(preliminary_clean, area_threshold=dynamic_noise_threshold)
            OriginalChosenMask = preliminary_clean.astype(np.uint8)
            mask_bool = OriginalChosenMask > 0
            cleaned_mask_bool = remove_small_objects(mask_bool, min_size=dynamic_noise_threshold * 3)
            BaseSilhouetteMask = cleaned_mask_bool.astype(np.uint8)
            
            if stencil_kernel_size % 2 == 0: stencil_kernel_size += 1
            stencil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stencil_kernel_size, stencil_kernel_size))

            Stencil = cv2.morphologyEx(BaseSilhouetteMask, cv2.MORPH_CLOSE, stencil_kernel)
            
            contours, _ = cv2.findContours(Stencil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                Stencil = np.zeros_like(Stencil)
                cv2.drawContours(Stencil, [main_contour], -1, color=1, thickness=cv2.FILLED)

            HoleMask = cv2.subtract(Stencil, BaseSilhouetteMask)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(HoleMask, 8, cv2.CV_32S)
            
            min_hole_area_threshold = int(np.sum(BaseSilhouetteMask) * 0.005) if np.sum(BaseSilhouetteMask) > 0 else 50
            
            FilteredHoleMask = np.zeros_like(HoleMask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_hole_area_threshold:
                    FilteredHoleMask[labels == i] = 1
            
            AccumulatedPatch = np.zeros_like(BaseSilhouetteMask)
            
            if 'complementary_masks' in best_base_data and np.any(FilteredHoleMask):
                if mask_geometry:
                    expand_size = max(3, int(min(mask_geometry['width'], mask_geometry['height']) * 0.04))
                else:
                    expand_size = 5
                
                if expand_size % 2 == 0: expand_size += 1
                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))

                for comp in best_base_data['complementary_masks']:
                    ComplementaryMask = comp['original_mask']
                    SeedPatch = cv2.bitwise_and(ComplementaryMask, FilteredHoleMask)
                    
                    if np.any(SeedPatch):
                        DilatedPatch = cv2.dilate(SeedPatch, expand_kernel, iterations=1)
                        FinalPatch = cv2.bitwise_and(DilatedPatch, FilteredHoleMask)
                        AccumulatedPatch = cv2.bitwise_or(AccumulatedPatch, FinalPatch)
                
          
            MergedMask = cv2.bitwise_or(BaseSilhouetteMask, AccumulatedPatch)
            if mask_geometry:
                bridge_size = max(5, int(min(mask_geometry['width'], mask_geometry['height']) * 0.05))
            else:
                bridge_size = 7
            if bridge_size % 2 == 0: bridge_size += 1
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
            
            BridgedMask = cv2.morphologyEx(MergedMask, cv2.MORPH_CLOSE, bridge_kernel)
            intermediate_mask = self._recompose_mask_with_detail_preservation(
                processed_mask=BridgedMask, 
                original_mask=OriginalChosenMask, 
                min_hole_area_threshold=min_hole_area_threshold,
                noise_threshold=dynamic_noise_threshold
            )
            
            final_mask_crop = self._final_mask_cleanup(intermediate_mask)

            self.sam_predictor.set_image(full_image, image_format='RGB')
            full_size_mask = np.zeros((fh, fw), dtype=np.float32)
            full_size_mask[cy1:cy2, cx1:cx2] = final_mask_crop.astype(np.float32)

            return [full_size_mask]

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in segmentation for face {face_index + 1}: {e}")
            if hasattr(self, 'full_image_for_sam'): 
                self.sam_predictor.set_image(self.full_image_for_sam, 'RGB')
            return []
        
    def _final_mask_cleanup(self, mask_np):

        if mask_np.sum() < 10:
            return mask_np

        mask_bool = mask_np > 0
        mask_area = np.sum(mask_bool)
        
        NOISE_AREA_PERCENT = 0.005
        ABSOLUTE_MIN_NOISE_PIXELS = 25
        
        noise_threshold = max(ABSOLUTE_MIN_NOISE_PIXELS, int(mask_area * NOISE_AREA_PERCENT))
        
        cleaned_mask_bool = remove_small_objects(mask_bool, min_size=noise_threshold)
        
        if np.sum(cleaned_mask_bool) == 0:
            return np.zeros_like(mask_np, dtype=np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask_bool.astype(np.uint8), 8, cv2.CV_32S)
        
        if num_labels > 1:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(component_areas)
            
            MIN_RELATIVE_SIZE = 0.02
            final_mask = np.zeros_like(cleaned_mask_bool)
            
            for i in range(1, num_labels):
                component_area = stats[i, cv2.CC_STAT_AREA]
                if component_area >= (largest_area * MIN_RELATIVE_SIZE):
                    final_mask[labels == i] = True
            
            cleaned_mask_bool = final_mask
        
        cleaned_mask_area = np.sum(cleaned_mask_bool)
        HOLE_AREA_PERCENT = 0.1
        hole_threshold = int(cleaned_mask_area * HOLE_AREA_PERCENT)
        
        final_mask_bool = remove_small_holes(cleaned_mask_bool, area_threshold=hole_threshold)
        
        return final_mask_bool.astype(np.uint8)
    def detect_faces(self, image_tensor, bbox_model_name, bbox_model_B_name, sam_model_name, sam_model_B_name, detection_confidence, sam_threshold, face_selection, attempt_face_completion=False):

        
        return self.detect_faces_with_optimal_combination_search(
            image_tensor=image_tensor,
            bbox_model_name=bbox_model_name, 
            bbox_model_B_name=bbox_model_B_name,
            seg_model_name=sam_model_name,
            seg_model_B_name=sam_model_B_name, 
            detection_confidence=detection_confidence,
            sam_threshold=sam_threshold,
            face_selection=face_selection,
            attempt_face_completion=attempt_face_completion
        )
    def _deduplicate_bbox_candidates(self, bbox_candidates):

        if len(bbox_candidates) <= 1:
            return bbox_candidates
        
        def calculate_iou(box1, box2):

            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        deduplicated = []
        used_indices = set()
        
        for i, candidate1 in enumerate(bbox_candidates):
            if i in used_indices:
                continue
                
            overlapping_group = [candidate1]
            used_indices.add(i)
            
            for j, candidate2 in enumerate(bbox_candidates[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                iou = calculate_iou(candidate1['bbox'], candidate2['bbox'])
                if iou > 0.3:
                    overlapping_group.append(candidate2)
                    used_indices.add(j)
            
            best_candidate = max(overlapping_group, key=lambda x: x['conf'])
            deduplicated.append(best_candidate)
            
            if len(overlapping_group) > 1:
                sources = [c['source'] for c in overlapping_group]
        
        return deduplicated
    
    def detect_faces_with_optimal_combination_search(self, image_tensor, bbox_model_name, bbox_model_B_name, seg_model_name, seg_model_B_name, detection_confidence, sam_threshold=0.5, face_selection=0, attempt_face_completion=False):

        try:
            check_for_interruption()
            
            all_detections, ground_truth_faces = self._detect_faces_phase1(
                image_tensor, bbox_model_name, bbox_model_B_name, detection_confidence
            )
            if not ground_truth_faces:
                return []
            
            seg_models = self._prepare_segmentation_models(seg_model_name, seg_model_B_name)
            if not seg_models:
                return [face['mask'] for face in ground_truth_faces[:1]]
            
            self.full_image_for_sam = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            fh, fw = self.full_image_for_sam.shape[:2]
            
            final_masks = []
            for face_idx, ground_truth_face in enumerate(ground_truth_faces):
                
                face_bbox_options = self._get_bbox_options_for_face(
                    ground_truth_face, all_detections
                )
                
                best_mask = self._optimize_face_combinations(
                    face_bbox_options, seg_models, fh, fw, face_idx
                )
                
                if best_mask is not None:
                    final_masks.append(best_mask)
                else:
                    final_masks.append(ground_truth_face['mask'])
            
            if face_selection == 0:
                return final_masks
            elif face_selection <= len(final_masks):
                return [final_masks[face_selection - 1]]
            else:
                return [final_masks[0]] if final_masks else []
                
        except Exception as e:
            print(f"Error in optimal combination search: {e}")
            return []
    def _detect_faces_phase1(self, image_tensor, bbox_model_name, bbox_model_B_name, detection_confidence):

        try:
            image_uint8 = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            h, w = image_uint8.shape[:2]
            pil_image = Image.fromarray(image_uint8, mode='RGB')
            
            all_detections = []
            
            if self.load_bbox_detector(bbox_model_name):
                results = self.bbox_model(pil_image, conf=detection_confidence, verbose=False)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                            conf = box.conf[0].item()
                            all_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'conf': conf,
                                'source': bbox_model_name
                            })
            
            if bbox_model_B_name and bbox_model_B_name != "None":
                if self.load_bbox_detector_B(bbox_model_B_name):
                    results = self.bbox_model_B(pil_image, conf=detection_confidence, verbose=False)
                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                                conf = box.conf[0].item()
                                all_detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'conf': conf,
                                    'source': bbox_model_B_name
                                })

            
            unique_faces = self._deduplicate_bbox_candidates(all_detections)
            
            ground_truth_faces = []
            for face in unique_faces:
                oval_mask = self._create_oval_mask(face['bbox'], h, w)
                ground_truth_faces.append({
                    'bbox': face['bbox'],
                    'mask': oval_mask,
                    'conf': face['conf'],
                    'source': face['source']
                })
            
            return all_detections, ground_truth_faces
            
        except Exception as e:
            print(f"Error in Phase 1 face detection: {e}")
            return [], []
        
    def _get_bbox_options_for_face(self, ground_truth_face, all_phase1_detections):

        try:
            gt_bbox = ground_truth_face['bbox']
            bbox_options = []
            
            def calculate_iou(box1, box2):
                x1_1, y1_1, x2_1, y2_1 = box1
                x1_2, y1_2, x2_2, y2_2 = box2
                
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                
                if x2_i <= x1_i or y2_i <= y1_i:
                    return 0.0
                
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            for detection in all_phase1_detections:
                iou = calculate_iou(gt_bbox, detection['bbox'])
                
                if iou > 0.1:
                    bbox_options.append({
                        'bbox': detection['bbox'],
                        'conf': detection['conf'],
                        'source': detection['source'],
                        'iou_with_gt': iou
                    })
            
            unique_bbox_options = []
            for option in bbox_options:
                is_duplicate = False
                for existing in unique_bbox_options:
                    if (option['bbox'] == existing['bbox'] and 
                        option['source'] == existing['source']):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_bbox_options.append(option)
            
            
            return unique_bbox_options
            
        except Exception as e:
            print(f"Error getting bbox options for face: {e}")
            return [{'bbox': ground_truth_face['bbox'], 'conf': ground_truth_face['conf'], 'source': 'fallback'}]
    
    def _optimize_face_combinations(self, bbox_options, seg_models, fh, fw, face_idx):

        try:
            all_combinations = []
            
            for bbox_option in bbox_options:
                bbox = bbox_option['bbox']
                oval_mask = self._create_oval_mask(bbox, fh, fw)
                
                face_data = {
                    'bbox': bbox,
                    'mask': oval_mask
                }
                
                for seg_model in seg_models:
                    try:
                        check_for_interruption()
                        
                        seg_loaded = False
                        if seg_model['type'] == 'yoloseg':
                            seg_loaded = self.load_yoloseg_model(seg_model['name'])
                        elif seg_model['type'] == 'sam':
                            seg_loaded = self.load_sam_model(seg_model['name'])
                        
                        if not seg_loaded:
                            print(f"    Failed to load {seg_model['name']}")
                            continue
                        
                        generated_masks = []
                        if seg_model['type'] == 'yoloseg':
                            mask_crop = self._generate_single_yoloseg_mask(face_data, fh, fw)
                            if mask_crop is not None:
                                full_mask = np.zeros((fh, fw), dtype=np.float32)
                                cx1, cy1, cx2, cy2 = self.make_crop_region(bbox, fw, fh)
                                full_mask[cy1:cy2, cx1:cx2] = mask_crop.astype(np.float32)
                                generated_masks = [full_mask]
                            else:
                                print(f"    YOLO-seg failed, using oval mask for scoring")
                                generated_masks = [face_data['mask']]
                                
                        elif seg_model['type'] == 'sam':
                            generated_masks = self.generate_sam_mask_for_face(face_data, face_idx, 0.5, False)
                        
                        for mask in generated_masks:
                            if np.sum(mask) > 50:
                                quality_score = self._calculate_mask_quality_score(mask, bbox)
                                
                                final_score = quality_score
                                
                                all_combinations.append({
                                    'mask': mask,
                                    'bbox_source': bbox_option['source'],
                                    'seg_source': seg_model['name'],
                                    'seg_type': seg_model['type'],
                                    'quality_score': quality_score,
                                    'final_score': final_score,
                                    'bbox_conf': bbox_option['conf']
                                })

                    
                    except Exception as e:
                        print(f"    Error testing {bbox_option['source']} + {seg_model['name']}: {e}")
                        continue
            
            if not all_combinations:
                return None
            
            best_combo = max(all_combinations, key=lambda x: x['final_score'])
            
            return best_combo['mask']
            
        except Exception as e:
            print(f"Error optimizing face combinations: {e}")
            return None
    

    def _generate_single_yoloseg_mask(self, face_data, fh, fw):

        try:
            bbox = face_data['bbox']
            cx1, cy1, cx2, cy2 = self.make_crop_region(bbox, fw, fh)
            crop_img = self.full_image_for_sam[cy1:cy2, cx1:cx2]
            crop_h, crop_w = crop_img.shape[:2]

            results = self.yoloseg_model(crop_img, conf=0.25, verbose=False)
            mask_crop = self._process_yoloseg_results(results, face_data, crop_w, crop_h)
            
            return mask_crop
            
        except Exception as e:
            print(f"Error generating YOLO-seg mask: {e}")
            return None

    def _calculate_mask_quality_score(self, mask, bbox_used_for_generation):

        try:
            if np.sum(mask) == 0:
                return 0.0
            
            x1, y1, x2, y2 = bbox_used_for_generation
            bbox_area = (x2 - x1) * (y2 - y1)
            
            h, w = mask.shape
            cx1, cy1, cx2, cy2 = self.make_crop_region(bbox_used_for_generation, w, h)
            
            if cy2 > cy1 and cx2 > cx1 and cy1 >= 0 and cx1 >= 0 and cy2 <= h and cx2 <= w:
                mask_crop = mask[cy1:cy2, cx1:cx2]
                tight_bbox_in_crop = [
                    max(0, x1-cx1), max(0, y1-cy1), 
                    min(cx2-cx1, x2-cx1), min(cy2-cy1, y2-cy1)
                ]
            else:
                print(f"      Invalid crop region, using full mask")
                mask_crop = mask
                tight_bbox_in_crop = [0, 0, w, h]
            
            mask_area_in_bbox = np.sum(mask_crop > 0)
            tight_bbox_area = (tight_bbox_in_crop[2] - tight_bbox_in_crop[0]) * (tight_bbox_in_crop[3] - tight_bbox_in_crop[1])
            coverage_ratio = mask_area_in_bbox / tight_bbox_area if tight_bbox_area > 0 else 0
            
            if 0.35 <= coverage_ratio <= 0.70:
                coverage_score = 1.0
            elif 0.20 <= coverage_ratio < 0.35:
                coverage_score = 0.2 + (coverage_ratio - 0.20) * 5.33
            elif 0.70 < coverage_ratio <= 0.85:
                coverage_score = 1.0 - (coverage_ratio - 0.70) * 1.33
            else:
                coverage_score = max(0.05, 0.2 - abs(coverage_ratio - 0.525) * 1.5)
            
            tx1, ty1, tx2, ty2 = tight_bbox_in_crop
            edge_buffer = 3
            boundary_violations = 0
            total_boundary_pixels = 0
            
            if ty1 + edge_buffer <= ty2 and tx1 < tx2:
                boundary_violations += np.sum(mask_crop[ty1:min(ty1+edge_buffer, ty2), tx1:tx2] > 0)
                total_boundary_pixels += (min(ty1+edge_buffer, ty2) - ty1) * (tx2 - tx1)
            
            if ty2 - edge_buffer >= ty1 and tx1 < tx2:
                boundary_violations += np.sum(mask_crop[max(ty1, ty2-edge_buffer):ty2, tx1:tx2] > 0)
                total_boundary_pixels += (ty2 - max(ty1, ty2-edge_buffer)) * (tx2 - tx1)
            
            if tx1 + edge_buffer <= tx2 and ty1 < ty2:
                boundary_violations += np.sum(mask_crop[ty1:ty2, tx1:min(tx1+edge_buffer, tx2)] > 0)
                total_boundary_pixels += (ty2 - ty1) * (min(tx1+edge_buffer, tx2) - tx1)
            
            if tx2 - edge_buffer >= tx1 and ty1 < ty2:
                boundary_violations += np.sum(mask_crop[ty1:ty2, max(tx1, tx2-edge_buffer):tx2] > 0)
                total_boundary_pixels += (ty2 - ty1) * (tx2 - max(tx1, tx2-edge_buffer))
            
            boundary_violation_ratio = boundary_violations / total_boundary_pixels if total_boundary_pixels > 0 else 0
            
            if boundary_violation_ratio <= 0.15:
                boundary_score = 1.0
            elif boundary_violation_ratio <= 0.35:
                boundary_score = 1.0 - (boundary_violation_ratio - 0.15) * 2.5
            else:
                boundary_score = max(0.1, 0.5 - (boundary_violation_ratio - 0.35) * 1.5)
            
            compactness_score = self.calculate_solidity_score(mask_crop)
            
            final_score = (coverage_score * 0.70 + boundary_score * 0.20 + compactness_score * 0.10)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error calculating mask quality: {e}")
            return 0.5

    def _create_oval_mask(self, bbox, h, w):

        x1, y1, x2, y2 = bbox
        oval_mask = np.zeros((h, w), dtype=np.float32)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(oval_mask, center, axes, 0, 0, 360, 1.0, -1)
        return oval_mask

    def _prepare_segmentation_models(self, seg_model_name, seg_model_B_name):

        seg_models = []
        
        if seg_model_name and not seg_model_name.endswith('None'):
            seg_models.append({
                'name': seg_model_name,
                'type': 'sam' if seg_model_name.endswith('.pth') else 'yoloseg',
                'is_primary': True
            })
        
        if seg_model_B_name and seg_model_B_name != "None" and seg_model_B_name != seg_model_name:
            seg_models.append({
                'name': seg_model_B_name,
                'type': 'sam' if seg_model_B_name.endswith('.pth') else 'yoloseg',
                'is_primary': False
            })
        
        return seg_models

    def detect_faces_bbox_only(self, image_tensor, bbox_model_name, bbox_model_B_name, detection_confidence):
        try:
            check_for_interruption()
            image_uint8 = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            h, w = image_uint8.shape[:2]
            pil_image = Image.fromarray(image_uint8, mode='RGB')

            if bbox_model_B_name is None or bbox_model_B_name == "None":
                if not self.load_bbox_detector(bbox_model_name): return []
                results = self.bbox_model(pil_image, conf=detection_confidence, verbose=False)
                if not results or results[0].boxes is None: return []
                
                detected_segments = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                    
                    oval_mask = np.zeros((h, w), dtype=np.float32)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(oval_mask, center, axes, 0, 0, 360, 1.0, -1)
                    detected_segments.append({'bbox': [x1, y1, x2, y2], 'mask': oval_mask})

                return detected_segments

            else:
                if not self.load_bbox_detector(bbox_model_name): return []
                if not self.load_bbox_detector_B(bbox_model_B_name): return []

                results_A = self.bbox_model(pil_image, conf=detection_confidence, verbose=False)
                results_B = self.bbox_model_B(pil_image, conf=detection_confidence, verbose=False)
                
                detected_segments = []

                model_A_segments = []
                if results_A and results_A[0].boxes is not None:
                    for box in results_A[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                            conf = box.conf[0].item()
                            oval_mask = np.zeros((h, w), dtype=np.float32)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                            cv2.ellipse(oval_mask, center, axes, 0, 0, 360, 1.0, -1)
                            model_A_segments.append({'bbox': [x1, y1, x2, y2], 'mask': oval_mask, 'conf': conf, 'source': 'A'})

                model_B_segments = []
                if results_B and results_B[0].boxes is not None:
                    for box in results_B[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                            conf = box.conf[0].item()
                            oval_mask = np.zeros((h, w), dtype=np.float32)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                            cv2.ellipse(oval_mask, center, axes, 0, 0, 360, 1.0, -1)
                            model_B_segments.append({'bbox': [x1, y1, x2, y2], 'mask': oval_mask, 'conf': conf, 'source': 'B'})

                all_segments = model_A_segments + model_B_segments
                if all_segments:
                    best_segment = max(all_segments, key=lambda x: x['conf'])
                    detected_segments = [{'bbox': best_segment['bbox'], 'mask': best_segment['mask']}]

                return detected_segments

        except Exception as e: 
            print(f"Error in BBOX detection: {e}")
            return []