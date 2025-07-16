import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .utils import check_for_interruption, find_model_path
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from PIL import Image

class ForbiddenVisionFaceDetector:
    def __init__(self):
        self.sam_model, self.sam_predictor, self.sam_model_name = None, None, None
        self.bbox_model, self.bbox_model_name = None, None

    def load_bbox_detector(self, model_name):
        try:
            if self.bbox_model is not None and self.bbox_model_name == model_name: return True
            model_path = find_model_path(model_name, 'yolo')
            if model_path is None: print(f"Error: BBOX (YOLO) model '{model_name}' not found."); return False
            from ultralytics import YOLO
            device = model_management.get_torch_device()
            if self.bbox_model is not None: del self.bbox_model
            self.bbox_model = YOLO(model_path); self.bbox_model.to(device)
            self.bbox_model_name = model_name
            print(f"Loaded BBOX model: {model_name}")
            return True
        except Exception as e: print(f"Error loading YOLO '{model_name}': {e}"); self.bbox_model = None; self.bbox_model_name = None; return False

    def load_sam_model(self, model_name):
        try:
            if self.sam_model is not None and self.sam_model_name == model_name: return True
            model_path = find_model_path(model_name, 'sam')
            if model_path is None: print(f"Error: SAM model '{model_name}' not found."); return False
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "default"
            if "vit_b" in model_name: model_type = "vit_b"
            elif "vit_l" in model_name: model_type = "vit_l"
            elif "vit_h" in model_name: model_type = "vit_h"
            device = model_management.get_torch_device()
            self.sam_model = sam_model_registry[model_type](checkpoint=model_path); self.sam_model.to(device)
            self.sam_predictor = SamPredictor(self.sam_model)
            self.sam_model_name = model_name
            print(f"Loaded SAM model: {model_name}")
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
                    pass  # Do nothing - don't fill this component
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
            final_mask_crop = self._recompose_mask_with_detail_preservation(
                processed_mask=BridgedMask, 
                original_mask=OriginalChosenMask, 
                min_hole_area_threshold=min_hole_area_threshold,
                noise_threshold=dynamic_noise_threshold
            )

            self.sam_predictor.set_image(full_image, image_format='RGB')
            full_size_mask = np.zeros((fh, fw), dtype=np.float32)
            full_size_mask[cy1:cy2, cx1:cx2] = final_mask_crop.astype(np.float32)
            final_mask_bool = final_mask_crop > 0
            final_mask_bool = remove_small_objects(final_mask_bool, min_size=dynamic_noise_threshold * 2)
            final_mask_crop = final_mask_bool.astype(np.float32)

            full_size_mask = np.zeros((fh, fw), dtype=np.float32)
            full_size_mask[cy1:cy2, cx1:cx2] = final_mask_crop
            return [full_size_mask]

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in segmentation for face {face_index + 1}: {e}")
            if hasattr(self, 'full_image_for_sam'): 
                self.sam_predictor.set_image(self.full_image_for_sam, 'RGB')
            return []

    def detect_faces_with_sam_refinement(self, image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection, attempt_face_completion=False):
        try:
            check_for_interruption()
            detected_faces = self.detect_faces_bbox_only(image_tensor, bbox_model_name, detection_confidence)
            if not detected_faces: return []
            
            if not self.load_sam_model(sam_model_name): 
                print("Warning: SAM model failed to load. Falling back to BBox masks.")
                return [face['mask'] for face in detected_faces if 'mask' in face]

            self.full_image_for_sam = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            all_final_masks = []
            for i, face_data in enumerate(detected_faces):
                check_for_interruption()
                generated_masks = self.generate_sam_mask_for_face(face_data, i, sam_threshold, attempt_face_completion)
                if generated_masks: all_final_masks.extend(generated_masks)
            
            if not all_final_masks:
                print("Warning: SAM refinement did not produce any masks. Falling back to BBox masks.")
                return [face['mask'] for face in detected_faces if 'mask' in face]
            
            if face_selection == 0: return all_final_masks
            elif face_selection <= len(all_final_masks): return [all_final_masks[face_selection - 1]]
            else: return [all_final_masks[0]]
        except model_management.InterruptProcessingException:
            raise
        except Exception as e: 
            print(f"Error in two-stage detection: {e}")
            return []

    def detect_faces(self, image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection, attempt_face_completion=False):
        if sam_model_name and sam_model_name != "None Found":
            return self.detect_faces_with_sam_refinement(image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection, attempt_face_completion)
        else:
            detected_segments = self.detect_faces_bbox_only(image_tensor, bbox_model_name, detection_confidence)
            masks = [seg['mask'] for seg in detected_segments if 'mask' in seg]
            if not masks: return []
            if face_selection == 0: return masks
            elif face_selection <= len(masks): return [masks[face_selection-1]]
            else: return [masks[0]]
    def detect_faces_bbox_only(self, image_tensor, bbox_model_name, detection_confidence):
        try:
            check_for_interruption()
            if not self.load_bbox_detector(bbox_model_name): return []
            image_uint8 = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            from PIL import Image
            h, w = image_uint8.shape[:2]
            results = self.bbox_model(Image.fromarray(image_uint8, mode='RGB'), conf=detection_confidence, verbose=False)
            if not results or results[0].boxes is None: return []
            
            detected_segments = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                rect_mask = np.zeros((h, w), dtype=np.float32); rect_mask[y1:y2, x1:x2] = 1.0
                detected_segments.append({'bbox': [x1, y1, x2, y2], 'mask': rect_mask})
            
            return detected_segments
        except Exception as e: 
            print(f"Error in BBOX detection: {e}")
            return []