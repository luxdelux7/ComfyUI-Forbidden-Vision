import torch
import numpy as np
import cv2
import os
import folder_paths
import comfy.model_management as model_management
from .utils import check_for_interruption, find_model_path
import time
from skimage.morphology import remove_small_holes, closing, disk
from PIL import Image, ImageDraw

class ForbiddenVisionFaceDetector:
    def __init__(self):
        from .utils import ensure_model_directories
        ensure_model_directories()
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

    def calculate_vertical_centering_score(self, mask_np, tight_bbox):
        try:
            if mask_np.sum() == 0: return 0.0
            coords = np.argwhere(mask_np > 0.5)
            if coords.shape[0] == 0: return 0.0
            mask_center_y = coords[:, 0].mean()
            
            tight_y1, tight_y2 = tight_bbox[1], tight_bbox[3]
            bbox_center_y = (tight_y1 + tight_y2) / 2.0
            
            bbox_height = tight_y2 - tight_y1
            if bbox_height == 0: return 0.5
            
            offset = (bbox_center_y - mask_center_y) / bbox_height
            
            return max(0.0, 1.0 - abs(offset) * 2.0)
        except Exception:
            return 0.0
        
    def calculate_boundary_fit_score(self, mask_np, tight_bbox):
        try:
            if mask_np.sum() == 0: return 1.0
            total_pixels = np.sum(mask_np > 0.5)
            if total_pixels == 0: return 1.0
            x1, y1, x2, y2 = tight_bbox
            h, w = mask_np.shape
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_mask[y1:y2, x1:x2] = 1
            inside_pixels = np.sum((mask_np > 0.5) & (bbox_mask > 0))
            return inside_pixels / total_pixels
        except Exception: return 0.0
    def calculate_edge_protrusion_score(self, mask_np, tight_bbox):
        try:
            if mask_np.sum() == 0: return 1.0
            x1, y1, x2, y2 = tight_bbox
            h, w = mask_np.shape
            
            edge_thickness_x = max(1, int((x2 - x1) * 0.05))
            edge_thickness_y = max(1, int((y2 - y1) * 0.05))
            
            left_edge = np.sum(mask_np[y1:y2, x1:x1+edge_thickness_x] > 0.5)
            right_edge = np.sum(mask_np[y1:y2, x2-edge_thickness_x:x2] > 0.5)
            top_edge = np.sum(mask_np[y1:y1+edge_thickness_y, x1:x2] > 0.5)
            bottom_edge = np.sum(mask_np[y2-edge_thickness_y:y2, x1:x2] > 0.5)
            
            total_edge_pixels = left_edge + right_edge + top_edge + bottom_edge
            max_left_right = edge_thickness_x * (y2 - y1) * 2
            max_top_bottom = edge_thickness_y * (x2 - x1) * 2
            max_edge_pixels = max_left_right + max_top_bottom
            
            if max_edge_pixels == 0: return 1.0
            
            edge_fill_ratio = total_edge_pixels / max_edge_pixels
            return max(0.0, 1.0 - edge_fill_ratio)
        except Exception: 
            return 0.5
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
    def cleanup_mask_intelligent(self, mask_np, area_ratio_threshold=0.05):
        try:
            if mask_np.sum() == 0: return mask_np
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, 4, cv2.CV_32S)
            if num_labels <= 2: return mask_np
            
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area_idx = areas.argmax() + 1
            max_area = areas[max_area_idx - 1]
            
            cleaned_mask = np.zeros_like(mask_uint8)
            cleaned_mask[labels == max_area_idx] = 1
            
            for i in range(1, num_labels):
                if i == max_area_idx: continue
                
                component_area = stats[i, cv2.CC_STAT_AREA]
                if component_area >= max_area * area_ratio_threshold:
                    component_mask = (labels == i)
                    dilated_main = cv2.dilate(cleaned_mask.astype(np.uint8), 
                                            np.ones((20, 20), np.uint8), iterations=1)
                    
                    if np.any(component_mask & dilated_main):
                        cleaned_mask[labels == i] = 1
                        
            return cleaned_mask.astype(np.float32)
        except Exception: return mask_np

    def make_crop_region(self, bbox, w, h, crop_factor=2.0):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        ew, eh = int(bw * (crop_factor-1.0)/2.0), int(bh * (crop_factor-1.0)/2.0)
        return [max(0, x1 - ew), max(0, y1 - eh), min(w, x2 + ew), min(h, y2 + eh)]
    def light_cleanup_for_scoring(self, mask_np, min_area=50):
        """Light cleanup just for scoring - remove tiny fragments."""
        try:
            if mask_np.sum() == 0: return mask_np
            
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 4, cv2.CV_32S)
            
            if num_labels <= 1: return mask_np
            
            cleaned_mask = np.zeros_like(mask_uint8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned_mask[labels == i] = 1
                    
            return cleaned_mask.astype(np.float32)
        except Exception:
            return mask_np
    def select_best_mask(self, all_masks_data, tight_bbox):

        print("--- Selecting Best Face Mask (Clean Face Prioritization) ---")
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
            
            if y1 > 0:
                outside_bbox_area += np.sum(mask[0:y1, :] > 0.5)
            
            if y2 < h:
                outside_bbox_area += np.sum(mask[y2:h, :] > 0.5)
            
            if x1 > 0:
                outside_bbox_area += np.sum(mask[max(0,y1):min(h,y2), 0:x1] > 0.5)
            
            if x2 < w:
                outside_bbox_area += np.sum(mask[max(0,y1):min(h,y2), x2:w] > 0.5)
            
            calculated_total = inside_bbox_area + outside_bbox_area
            if abs(calculated_total - total_mask_area) > 5:
                print(f"  WARNING: Containment calculation mismatch for Mask {data['index']}")
                print(f"    Total: {total_mask_area}, Inside: {inside_bbox_area}, Outside: {outside_bbox_area}")
            
            containment_ratio = inside_bbox_area / total_mask_area if total_mask_area > 0 else 0
            coverage_ratio = inside_bbox_area / bbox_area if bbox_area > 0 else 0
            
            print(f"  Mask {data['index']} containment check: {outside_bbox_area}/{total_mask_area} outside = {(outside_bbox_area/total_mask_area)*100:.1f}% outside")
            
            outside_ratio = outside_bbox_area / total_mask_area if total_mask_area > 0 else 0
            
            if outside_ratio <= 0.02:
                containment_score = 1.0
            elif outside_ratio <= 0.05:
                containment_score = 0.8
            elif outside_ratio <= 0.10:
                containment_score = 0.6
            elif outside_ratio <= 0.15:
                containment_score = 0.4
            else:
                containment_score = 0.2
            
            coverage_score = self.calculate_face_coverage_ratio(mask, tight_bbox)
            mask_uint8 = (mask > 0.5).astype(np.uint8)
            
            merge_kernel_size = max(5, int(bbox_width * 0.07))
            if merge_kernel_size % 2 == 0: merge_kernel_size += 1
            merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel_size, merge_kernel_size))
            
            mask_for_cleanliness_check = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, merge_kernel)
            num_labels_orig, _, _, _ = cv2.connectedComponentsWithStats(mask_for_cleanliness_check, 4, cv2.CV_32S)
            
            cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_for_analysis = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, cleanup_kernel)
            cleaned_for_analysis = cv2.morphologyEx(cleaned_for_analysis, cv2.MORPH_CLOSE, cleanup_kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_for_analysis, 4, cv2.CV_32S)
            
            edge_noise = 0
            edge_thickness = 2
            
            if y1 + edge_thickness < y2:
                edge_noise += np.sum(mask[y1:y1+edge_thickness, x1:x2] > 0.5)
            
            if y2 - edge_thickness > y1:
                edge_noise += np.sum(mask[y2-edge_thickness:y2, x1:x2] > 0.5)
            
            if x1 + edge_thickness < x2:
                edge_noise += np.sum(mask[y1:y2, x1:x1+edge_thickness] > 0.5)
            
            if x2 - edge_thickness > x1:
                edge_noise += np.sum(mask[y1:y2, x2-edge_thickness:x2] > 0.5)
            
            edge_noise_ratio = edge_noise / total_mask_area if total_mask_area > 0 else 0
            
            if num_labels_orig <= 5 and edge_noise_ratio < 0.05:
                cleanliness_score = 1.0
            elif num_labels_orig <= 10 and edge_noise_ratio < 0.10:
                cleanliness_score = 0.8
            elif num_labels_orig <= 20 and edge_noise_ratio < 0.15:
                cleanliness_score = 0.6
            elif num_labels_orig <= 40 and edge_noise_ratio < 0.25:
                cleanliness_score = 0.4
            else:
                cleanliness_score = 0.2
            
            if outside_bbox_area == 0 and edge_noise_ratio == 0:
                cleanliness_score = min(1.0, cleanliness_score + 0.2)
            
            if num_labels_orig > 60:
                cleanliness_score *= 0.7
            
            print(f"  Mask {data['index']} cleanliness: {num_labels_orig-1} components (post-merge), {edge_noise_ratio*100:.1f}% edge noise -> score {cleanliness_score:.2f}")
            
            shape_score = self.calculate_comprehensive_shape_score(mask)
            
            final_score = (containment_score * 0.50 +  # <-- 50%
                        coverage_score * 0.10 +  # <-- 10%
                        cleanliness_score * 0.30 +
                        shape_score * 0.10)
            
            if outside_bbox_area == 0:
                final_score += 0.04
            
            if edge_noise_ratio < 0.01:
                final_score += 0.02
   
            
            sam_confidence_bonus = data.get('sam_score', 0) * 0.01
            final_score += sam_confidence_bonus
            
            data['final_score'] = final_score
            data['containment_ratio'] = containment_ratio
            data['coverage_ratio'] = coverage_ratio
            data['cleanliness_score'] = cleanliness_score
            scored_candidates.append(data)

            perfect_containment_bonus = 0.04 if outside_bbox_area == 0 else 0
            clean_edges_bonus = 0.02 if edge_noise_ratio < 0.01 else 0
            sam_bonus = data.get('sam_score', 0) * 0.01
            total_bonus = perfect_containment_bonus + clean_edges_bonus + sam_bonus
            
            print(f"  Mask {data['index']}: Final Score={final_score:.3f} | "
                f"Containment={containment_ratio:.2f} ({containment_score:.2f}), "
                f"Coverage={coverage_ratio:.2f} ({coverage_score:.2f}), "
                f"Clean={cleanliness_score:.2f}, Shape={shape_score:.2f}, "
                f"Bonus={total_bonus:.3f}")
        
        if not scored_candidates:
            print("ERROR: No candidates could be scored. Cannot select a mask.")
            return None

        best_mask_data = max(scored_candidates, key=lambda x: x['final_score'])

        print(f"--- Selected Mask #{best_mask_data['index']} with score {best_mask_data['final_score']:.3f} ---")
        
        best_base = best_mask_data
        best_base['complementary_masks'] = [
            d for d in all_masks_data 
            if d['index'] != best_base['index']
        ]
        
        return best_base
    def identify_facial_holes(self, mask_np, mask_geometry, zones):
        try:
            if mask_np.sum() == 0 or mask_geometry is None or zones is None:
                return []
                
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            safe_boundary = zones['safe_fill_boundary']
            
            inverted_mask = cv2.bitwise_not(mask_uint8 * 255)
            
            h, w = inverted_mask.shape[:2]
            if h <= 2 or w <= 2:
                return []
                
            mask_for_flooding = inverted_mask.copy()
            mask_flood = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_for_flooding, mask_flood, (0, 0), 0)
            
            holes_mask = mask_for_flooding.astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes_mask, 4, cv2.CV_32S)
            
            legitimate_holes = []
            mask_area = mask_geometry['area']
            centroid_x, centroid_y = zones['centroid']
            
            print(f"  Found {num_labels-1} potential holes in mask-based analysis")
            
            for i in range(1, num_labels):
                hole_area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = centroids[i]
                
                if hole_area < 5:
                    continue
                    
                if safe_boundary[int(cy), int(cx)] == 0:
                    print(f"    Hole {i}: outside safe boundary at ({cx:.1f}, {cy:.1f}) - rejected")
                    continue
                
                distance_from_center = np.sqrt((cx - centroid_x)**2 + (cy - centroid_y)**2)
                max_distance = min(mask_geometry['width'], mask_geometry['height']) * 0.4
                
                if distance_from_center > max_distance:
                    print(f"    Hole {i}: too far from face center ({distance_from_center:.1f} > {max_distance:.1f}) - rejected")
                    continue
                
                relative_hole_size = hole_area / mask_area if mask_area > 0 else 0
                if relative_hole_size > 0.15:
                    print(f"    Hole {i}: too large ({relative_hole_size:.3f} of mask area) - rejected")
                    continue
                
                hole_w = stats[i, cv2.CC_STAT_WIDTH]
                hole_h = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = hole_w / hole_h if hole_h > 0 else 1
                
                if not (0.1 < aspect_ratio < 10.0):
                    print(f"    Hole {i}: bad aspect ratio ({aspect_ratio:.2f}) - rejected")
                    continue
                
                hole_type = 'unknown'
                if zones['eye_zone'][int(cy), int(cx)] == 1:
                    hole_type = 'eye'
                elif zones['nose_zone'][int(cy), int(cx)] == 1:
                    hole_type = 'nose'
                elif zones['mouth_zone'][int(cy), int(cx)] == 1:
                    hole_type = 'mouth'
                else:
                    print(f"    Hole {i}: not in recognized facial zone - rejected")
                    continue
                
                legitimate_holes.append({
                    'label': i,
                    'area': hole_area,
                    'centroid': (cx, cy),
                    'type': hole_type,
                    'aspect_ratio': aspect_ratio,
                    'relative_size': relative_hole_size,
                    'distance_from_center': distance_from_center
                })
                print(f"      -> Accepted as {hole_type} (area: {hole_area} pixels, distance: {distance_from_center:.1f})")
            
            print(f"  Identified {len(legitimate_holes)} facial holes: {[h['type'] for h in legitimate_holes]}")
            return legitimate_holes
            
        except Exception as e:
            print(f"Error identifying facial holes: {e}")
            import traceback
            traceback.print_exc()
            return []
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
        
        print(f"  Generated a pool of {len(all_masks)} masks from dual-prompt strategy.")
        for i, (m, s) in enumerate(zip(all_masks, all_scores)):
            self.debug_save_sam_mask(m, f"RAW_CANDIDATE_Idx{i}_Score{s:.3f}", bbox, s)
        return all_masks, all_scores
    
    def refine_mask_by_grafting(self, best_mask_data, tight_bbox):
        try:
            if not best_mask_data or 'mask' not in best_mask_data:
                return np.zeros((100, 100), dtype=np.float32)

            base_mask = (best_mask_data['mask'] > 0.5).astype(np.uint8)
            h, w = base_mask.shape
            
            print(f"--- Refining mask #{best_mask_data['index']} with corrected hybrid refinement ---")

            mask_geometry = self.analyze_mask_geometry(base_mask.astype(np.float32))
            if mask_geometry is None:
                print("  Could not analyze mask geometry, returning original mask")
                return base_mask.astype(np.float32)

            mask_width = mask_geometry['width']
            mask_height = mask_geometry['height']

            prelim_kernel_size = max(5, int(min(mask_width, mask_height) * 0.05))
            if prelim_kernel_size % 2 == 0: prelim_kernel_size += 1
            
            prelim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prelim_kernel_size, prelim_kernel_size))
            smoothed_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, prelim_kernel)
            
            pixels_added_smoothing = int(np.sum(smoothed_mask)) - int(np.sum(base_mask))
            print(f"  Step 1 (Smoothing): Added {pixels_added_smoothing} pixels with kernel size {prelim_kernel_size}.")
            
            contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            closed_mask = remove_small_holes(smoothed_mask.astype(bool), area_threshold=500)

            # Bridge narrow internal gaps (without growing whole mask)
            bridged_mask = closing(closed_mask, footprint=disk(10))

            # Convert to uint8 for OpenCV
            bridged_mask_uint8 = (bridged_mask > 0).astype(np.uint8)

            # Find filled contour
            contours, _ = cv2.findContours(bridged_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("  WARNING: No contours after bridging. Fallback to original mask.")
                return base_mask.astype(np.float32)

            main_contour = max(contours, key=cv2.contourArea)

            # Sanity check like before
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(main_contour)

            solidity = contour_area / hull_area if hull_area > 0 else 0
            image_area = h * w
            area_ratio = contour_area / image_area if image_area > 0 else 0

            if solidity < 0.85 and area_ratio > 0.8:
                print(f"  WARNING: Refinement failed sanity check (solidity: {solidity:.2f}, area: {area_ratio:.2f}). Falling back.")
                return base_mask.astype(np.float32)

            # Final filled mask
            final_filled_mask = np.zeros_like(bridged_mask_uint8)
            cv2.drawContours(final_filled_mask, [main_contour], -1, color=1, thickness=cv2.FILLED)

            pixels_added_filling = int(np.sum(final_filled_mask)) - int(np.sum(smoothed_mask))
            print(f"  Step 2 (Hole Fill): Added {pixels_added_filling} pixels by filling main contour.")

            original_pixels = int(np.sum(base_mask.astype(np.int64)))
            final_pixels = int(np.sum(final_filled_mask.astype(np.int64)))
            print(f"  Refinement complete. Original pixels: {original_pixels}, Final pixels: {final_pixels}")

            return final_filled_mask.astype(np.float32)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during mask refinement: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                if 'best_mask_data' in locals() and 'mask' in best_mask_data:
                    return (best_mask_data['mask'] > 0.5).astype(np.float32)
                return np.zeros((100, 100), dtype=np.float32)
            except:
                return np.zeros((100, 100), dtype=np.float32)
    def calculate_comprehensive_shape_score(self, mask_np):
        try:
            if mask_np.sum() == 0: return 0.0
            
            # Metric 1: Compactness
            coords = np.argwhere(mask_np > 0.5)
            if coords.shape[0] < 100: return 0.0
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            mask_bbox_area = (max_x - min_x) * (max_y - min_y)
            mask_pixel_area = coords.shape[0]
            compactness_score = mask_pixel_area / mask_bbox_area if mask_bbox_area > 0 else 0

            # Metric 2: Circularity
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
    def create_mask_based_zones(self, mask_np, mask_geometry):
        try:
            if mask_geometry is None:
                return None
                
            h, w = mask_np.shape
            centroid_x, centroid_y = mask_geometry['centroid']
            min_x, min_y, max_x, max_y = mask_geometry['bbox']
            mask_height = mask_geometry['height']
            mask_width = mask_geometry['width']
            
            zones = {}
            
            upper_third_y = min_y + int(mask_height * 0.33)
            lower_third_y = min_y + int(mask_height * 0.67)
            
            eye_zone = np.zeros((h, w), dtype=np.uint8)
            eye_zone[min_y:upper_third_y, min_x:max_x] = 1
            zones['eye_zone'] = eye_zone
            
            nose_zone = np.zeros((h, w), dtype=np.uint8)
            nose_zone[upper_third_y:lower_third_y, min_x:max_x] = 1
            zones['nose_zone'] = nose_zone
            
            mouth_zone = np.zeros((h, w), dtype=np.uint8)
            mouth_zone[lower_third_y:max_y, min_x:max_x] = 1
            zones['mouth_zone'] = mouth_zone
            
            core_face_kernel_size = max(5, int(min(mask_width, mask_height) * 0.1))
            if core_face_kernel_size % 2 == 0:
                core_face_kernel_size += 1
            core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_face_kernel_size, core_face_kernel_size))
            
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            core_face_region = cv2.erode(mask_uint8, core_kernel, iterations=1)
            zones['core_face'] = core_face_region
            
            safe_fill_kernel_size = max(3, int(min(mask_width, mask_height) * 0.05))
            if safe_fill_kernel_size % 2 == 0:
                safe_fill_kernel_size += 1
            safe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_fill_kernel_size, safe_fill_kernel_size))

            safe_fill_boundary = cv2.dilate(mask_uint8, safe_kernel, iterations=1)
            zones['safe_fill_boundary'] = safe_fill_boundary
            
            zones['centroid'] = (centroid_x, centroid_y)
            zones['mask_bounds'] = (min_x, min_y, max_x, max_y)
            
            return zones
            
        except Exception as e:
            print(f"Error creating mask-based zones: {e}")
            return None

    
    def generate_sam_mask_for_face(self, face_data, face_index):
        try:
            check_for_interruption()
            print(f"--- Processing Face {face_index + 1} using 'Silhouette and Filler' strategy ---")

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
                cleaned_mask_for_scoring = self.light_cleanup_for_scoring(m.astype(np.float32), min_area=50)
                if np.sum(cleaned_mask_for_scoring) == 0: continue
                all_masks_data.append({
                    'index': i,
                    'mask': cleaned_mask_for_scoring,
                    'original_mask': (m > 0.5).astype(np.uint8),
                    'sam_score': s
                })

            best_base_data = self.select_best_mask(all_masks_data, crop_bbox)

            if not best_base_data:
                print("Warning: No suitable base mask was selected. Returning empty mask.")
                self.sam_predictor.set_image(full_image, image_format='RGB')
                return []
            
            BaseSilhouetteMask = (best_base_data['mask'] > 0.5).astype(np.uint8)
            
            print("  Refining with Silhouette and Filler strategy (using Morphological Closing)...")

            # --- REPLACEMENT FOR CONVEX HULL ---
            # Use Morphological Closing instead of Convex Hull to create a less aggressive Stencil
            # that preserves the main contours while closing holes.
            mask_geometry = self.analyze_mask_geometry(BaseSilhouetteMask.astype(np.float32))
            if mask_geometry:
                # Use a large kernel, ~15-20% of the mask's smaller dimension, to close large holes like eye sockets
                stencil_kernel_size = max(15, int(min(mask_geometry['width'], mask_geometry['height']) * 0.20))
            else:
                stencil_kernel_size = 25 # Fallback size
            
            if stencil_kernel_size % 2 == 0: stencil_kernel_size += 1
            stencil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stencil_kernel_size, stencil_kernel_size))
            print(f"  Creating Stencil with closing kernel of size {stencil_kernel_size}x{stencil_kernel_size}.")
            Stencil = cv2.morphologyEx(BaseSilhouetteMask, cv2.MORPH_CLOSE, stencil_kernel)
            # --- END OF REPLACEMENT ---

            HoleMask = cv2.subtract(Stencil, BaseSilhouetteMask)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(HoleMask, 8, cv2.CV_32S)
            
            min_hole_area = int(np.sum(BaseSilhouetteMask) * 0.005) 
            print(f"  Filtering holes: only filling holes larger than {min_hole_area} pixels.")
            
            FilteredHoleMask = np.zeros_like(HoleMask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_hole_area:
                    FilteredHoleMask[labels == i] = 1
            
            print(f"  Identified {np.sum(FilteredHoleMask > 0)} pixels in {cv2.connectedComponents(FilteredHoleMask)[0] - 1} significant holes to be filled.")

            AccumulatedPatch = np.zeros_like(BaseSilhouetteMask)
            
            if 'complementary_masks' in best_base_data and np.any(FilteredHoleMask):
                if mask_geometry:
                    expand_size = max(3, int(min(mask_geometry['width'], mask_geometry['height']) * 0.04))
                else:
                    expand_size = 5
                
                if expand_size % 2 == 0: expand_size += 1
                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
                print(f"  Using expansion kernel of size {expand_size}x{expand_size} for grafting.")

                for comp in best_base_data['complementary_masks']:
                    ComplementaryMask = comp['original_mask']
                    SeedPatch = cv2.bitwise_and(ComplementaryMask, FilteredHoleMask)
                    
                    if np.any(SeedPatch):
                        DilatedPatch = cv2.dilate(SeedPatch, expand_kernel, iterations=1)
                        FinalPatch = cv2.bitwise_and(DilatedPatch, FilteredHoleMask)
                        AccumulatedPatch = cv2.bitwise_or(AccumulatedPatch, FinalPatch)
                
                print(f"  Accumulated {np.sum(AccumulatedPatch > 0)} pixels from {len(best_base_data['complementary_masks'])} complementary masks.")

            MergedMask = cv2.bitwise_or(BaseSilhouetteMask, AccumulatedPatch)
            
            if mask_geometry:
                bridge_size = max(5, int(min(mask_geometry['width'], mask_geometry['height']) * 0.05))
            else:
                bridge_size = 7
            if bridge_size % 2 == 0: bridge_size += 1
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
            print(f"  Bridging disconnected components with kernel of size {bridge_size}x{bridge_size}.")
            BridgedMask = cv2.morphologyEx(MergedMask, cv2.MORPH_CLOSE, bridge_kernel)

            final_contours, _ = cv2.findContours(BridgedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not final_contours:
                print("  Warning: Merged mask has no contours. Returning bridged mask as is.")
                final_mask_crop = BridgedMask
            else:
                final_main_contour = max(final_contours, key=cv2.contourArea)
                final_mask_crop = np.zeros_like(BridgedMask)
                cv2.drawContours(final_mask_crop, [final_main_contour], -1, color=1, thickness=cv2.FILLED)
                print("  Final mask polished and holes filled.")

            self.sam_predictor.set_image(full_image, image_format='RGB')
            full_size_mask = np.zeros((fh, fw), dtype=np.float32)
            full_size_mask[cy1:cy2, cx1:cx2] = final_mask_crop.astype(np.float32)
            
            self.debug_save_sam_mask(full_size_mask, f"Face{face_index + 1}_FINAL_MASK_REFINED", bbox)
            return [full_size_mask]

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in segmentation for face {face_index + 1}: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'full_image_for_sam'): 
                self.sam_predictor.set_image(self.full_image_for_sam, 'RGB')
            return []

    def generate_sam_mask_for_segment(self, seg_data, seg_index):
        try:
            check_for_interruption()
            print(f"--- Processing Segment {seg_index} ---")

            bbox = seg_data['bbox']
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
                data = {'mask': m.astype(bool)}
                total_mask_area = np.sum(data['mask'])
                if total_mask_area == 0: continue
                
                data['solidity'] = self.calculate_solidity_score(data['mask'])
                
                x1, y1, x2, y2 = crop_bbox
                inside_bbox_area = np.sum(data['mask'][y1:y2, x1:x2])
                data['outside_ratio'] = (total_mask_area - inside_bbox_area) / total_mask_area
                all_masks_data.append(data)

            good_candidates = [
                d['mask'] for d in all_masks_data
                if d['solidity'] > 0.80 and d['outside_ratio'] < 0.15
            ]

            if not good_candidates:
                print("No 'excellent' masks found. Relaxing filter criteria for complex/stylized faces...")
                good_candidates = [
                    d['mask'] for d in all_masks_data
                    if d['solidity'] > 0.60 and d['outside_ratio'] < 0.30
                ]

            if not good_candidates:
                print("Warning: No usable candidate masks found even after relaxing criteria. Returning empty mask.")
                self.sam_predictor.set_image(full_image, image_format='RGB')
                return []

            print(f"Found {len(good_candidates)} candidates to build consensus from.")

            union_mask = np.zeros_like(good_candidates[0], dtype=bool)
            for mask in good_candidates: union_mask = np.logical_or(union_mask, mask)

            intersection_mask = np.ones_like(good_candidates[0], dtype=bool)
            for mask in good_candidates: intersection_mask = np.logical_and(intersection_mask, mask)
            
            mask_width = crop_bbox[2] - crop_bbox[0]
            dilation_size = max(5, int(mask_width * 0.08))
            if dilation_size % 2 == 0: dilation_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            dilated_core = cv2.dilate(intersection_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            combined_mask = np.logical_and(dilated_core, union_mask)
            
            polish_kernel_size = max(3, int(mask_width * 0.02))
            if polish_kernel_size % 2 == 0: polish_kernel_size += 1
            polish_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (polish_kernel_size, polish_kernel_size))
            final_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, polish_kernel)

            self.sam_predictor.set_image(full_image, image_format='RGB')
            full_size_mask = np.zeros((fh, fw), dtype=np.float32)
            full_size_mask[cy1:cy2, cx1:cx2] = final_mask.astype(np.float32)
            
            self.debug_save_sam_mask(full_size_mask, f"Seg{seg_index}_FINAL_MASK", bbox)
            return [full_size_mask]

        except Exception as e:
            print(f"Error in segmentation: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'full_image_for_sam'): 
                self.sam_predictor.set_image(self.full_image_for_sam, 'RGB')
            return []
    def refine_face_contour(self, mask_np, tight_bbox):
        try:
            if mask_np.sum() == 0: return mask_np
            
            mask_uint8 = (mask_np > 0.5).astype(np.uint8)
            
            contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return mask_np
            
            x1, y1, x2, y2 = tight_bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 0.4 * bbox_area <= area <= 0.9 * bbox_area:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        bbox_cx = (x1 + x2) / 2
                        bbox_cy = (y1 + y2) / 2
                        
                        if (abs(cx - bbox_cx) < 0.25 * (x2 - x1) and 
                            abs(cy - bbox_cy) < 0.25 * (y2 - y1)):
                            valid_contours.append((contour, area))
            
            if not valid_contours:
                largest_contour = max(contours, key=cv2.contourArea)
                valid_contours = [(largest_contour, cv2.contourArea(largest_contour))]
            
            main_contour = max(valid_contours, key=lambda x: x[1])[0]
            
            refined_mask = np.zeros_like(mask_uint8)
            
            epsilon = 0.005 * cv2.arcLength(main_contour, True)
            smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)
            
            cv2.fillPoly(refined_mask, [smoothed_contour], 1)
            
            return refined_mask.astype(np.float32)
        except Exception as e:
            print(f"Error in contour refinement: {e}")
            return mask_np
    def detect_faces_with_sam_refinement(self, image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection):
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
                generated_masks = self.generate_sam_mask_for_face(face_data, i)
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
            import traceback
            traceback.print_exc()
            return []

    def detect_faces(self, image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection):
        if sam_model_name and sam_model_name != "None Found":
            return self.detect_faces_with_sam_refinement(image_tensor, bbox_model_name, sam_model_name, detection_confidence, sam_threshold, face_selection)
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
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                rect_mask = np.zeros((h, w), dtype=np.float32); rect_mask[y1:y2, x1:x2] = 1.0
                detected_segments.append({'bbox': [x1, y1, x2, y2], 'mask': rect_mask})
            
            self.debug_save_yolo_detection(image_uint8, detected_segments, "Stage1_Detection")
            return detected_segments
        except Exception as e: 
            print(f"Error in BBOX detection: {e}")
            import traceback; traceback.print_exc()
            return []
    def debug_save_yolo_detection(self, image_uint8, detected_segments, stage_name):
        try:
            debug_dir = os.path.join(folder_paths.get_output_directory(), "debug_yolo_detection")
            os.makedirs(debug_dir, exist_ok=True)
            img = Image.fromarray(image_uint8); draw = ImageDraw.Draw(img)
            for seg in detected_segments: draw.rectangle(seg['bbox'], outline=(0, 255, 0), width=4)
            img.save(os.path.join(debug_dir, f"{int(time.time() * 1000)}_{stage_name}.png"))
        except Exception as e: print(f"Error saving YOLO debug: {e}")

    def debug_save_sam_mask(self, mask, filename_suffix, bbox=None, score=None):
        try:
            debug_dir = os.path.join(folder_paths.get_output_directory(), "debug_sam_masks")
            os.makedirs(debug_dir, exist_ok=True)
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            mask_img = Image.fromarray((np.clip(mask_np.squeeze(),0,1)*255).astype(np.uint8), 'L').convert("RGB")
            draw = ImageDraw.Draw(mask_img)
            if bbox: draw.rectangle(list(map(int, bbox)), outline=(255,0,0), width=2)
            if score is not None: draw.text((10, 10), f"Score: {score:.3f}", fill=(0, 255, 0))
            mask_img.save(os.path.join(debug_dir, f"{int(time.time() * 1000)}_{filename_suffix}.png"))
        except Exception as e: print(f"Error saving debug SAM mask: {e}")