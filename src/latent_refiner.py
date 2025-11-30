import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import nodes
import kornia
import os
import urllib.request
from .utils import check_for_interruption, get_refiner_upscaler_models, clean_model_name, DepthAnythingManager

class LatentRefiner:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_refiner_upscaler_models()
        default_upscaler = upscaler_models[1] if len(upscaler_models) > 1 else upscaler_models[0]        

        return {
            "required": {
                "enable_upscale": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "upscale_model": (upscaler_models, {"default": default_upscaler}),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.05}),
                
                "enable_auto_tone": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "auto_tone_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),

                "ai_colors": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "ai_colors_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                "ai_details": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "ai_details_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                
                "ai_relight": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "ai_relight_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),

                "ai_enable_dof": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "ai_dof_strength": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}), 
                "ai_dof_focus_depth": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.99, "step": 0.01}),

                "ai_depth_model": (["V2-Small", "V2-Base"], {"default": "V2-Small"}),
                
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            },
            "optional": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "refined_image_preview")
    FUNCTION = "refine_and_process"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        self.upscaler_model = None
        self.upscaler_model_name = None
        self.depth_manager = DepthAnythingManager.get_instance()
        self.cached_input_hash_depth = None
        self._invalidate_cache()

    def _invalidate_cache(self):
        print("Refiner: Invalidating all caches.")
        self.cached_depth_map = None
        self.cached_decoded_image = None
        self.cached_vae_hash = None
        self.cached_input_hash = None
        self.cached_input_hash_depth = None
        self.cached_depth_model_name = None

    def _get_vae_hash(self, vae):
        if vae is None:
            return None
        try:
            return hash((id(vae), str(vae.device) if hasattr(vae, 'device') else 'unknown'))
        except Exception:
            return id(vae)
    def _get_tensor_hash(self, tensor):
        if tensor is None:
            return None
        try:
            return hash((
                tensor.shape,
                tensor.dtype,
                tensor.device,
                float(tensor.sum().item()),
                float(tensor.mean().item())
            ))
        except Exception:
            return hash((tensor.shape, tensor.dtype, str(tensor.device)))
    
    def _run_and_cache_analysis(self, image_tensor, run_depth, depth_model_name="V2-Small"):
    
        try:
            check_for_interruption()
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            device = image_tensor.device
            
            if run_depth:
                self.cached_depth_map = None
            
            img_np_uint8 = None
            
            if run_depth:
                if img_np_uint8 is None:
                    img_np_uint8 = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                
                depth_tensor = self.depth_manager.infer_depth_full(img_np_uint8, depth_model_name)
                
                if depth_tensor is not None:
                    if depth_tensor.shape[-2:] != (h, w):
                        depth_tensor_resized = F.interpolate(
                            depth_tensor, size=(h, w), mode="bilinear", align_corners=False
                        )
                        self.cached_depth_map = depth_tensor_resized.to(device)
                    else:
                        self.cached_depth_map = depth_tensor.to(device)
                else:
                    print("Refiner: Depth inference failed, skipping depth estimation.")
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: An error occurred during AI analysis: {e}")
            if run_depth:
                self.cached_input_hash_depth = None
        
    def load_upscaler_model(self, model_name):
        clean_model_name_val = clean_model_name(model_name)
        
        if self.upscaler_model is not None and self.upscaler_model_name == clean_model_name_val:
            return self.upscaler_model
        try:
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            self.upscaler_model = upscaler_loader.load_model(clean_model_name_val)[0]
            self.upscaler_model_name = clean_model_name_val
            return self.upscaler_model
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error loading upscaler model {clean_model_name_val}: {e}")
            self.upscaler_model = None; self.upscaler_model_name = None
            return None
    
    
    def refine_and_process(self,
                        enable_auto_tone,
                        auto_tone_strength,
                        enable_upscale, upscale_model, upscale_factor,
                        ai_colors, ai_colors_strength,
                        ai_details, ai_details_strength,
                        ai_relight, ai_relight_strength,
                        ai_enable_dof, ai_dof_strength, ai_dof_focus_depth,
                        ai_depth_model,
                        use_tiled_vae, tile_size,
                        latent=None, vae=None, image=None, **kwargs):
        try:
            check_for_interruption()
            device = model_management.get_torch_device()

            is_latent_input = latent is not None and "samples" in latent
            is_image_input = image is not None

            if not is_latent_input and not is_image_input:
                print("Warning: No valid inputs provided.")
                return ({"samples": torch.zeros((1, 4, 64, 64))}, torch.zeros((1, 64, 64, 3)))

            is_dof_active = ai_enable_dof
            is_depth_needed = is_dof_active or (ai_details and ai_details_strength > 0) or (ai_relight and ai_relight_strength > 0)

            input_key_tensor = latent["samples"] if is_latent_input else image
            current_input_hash = self._get_tensor_hash(input_key_tensor)

            decoded_image = None
            if is_image_input:
                decoded_image = image.to(device)
            elif is_latent_input and vae is not None:
                current_vae_hash = self._get_vae_hash(vae)
                can_use_vae_cache = (
                    self.cached_decoded_image is not None and
                    self.cached_vae_hash == current_vae_hash and
                    self.cached_input_hash == current_input_hash
                )
                if can_use_vae_cache:
                    decoded_image = self.cached_decoded_image
                else:
                    decoded_image = vae.decode(input_key_tensor.to(device))
                    self.cached_decoded_image = decoded_image
                    self.cached_vae_hash = current_vae_hash
                    self.cached_input_hash = current_input_hash


            if decoded_image is None:
                dummy_latent = latent if is_latent_input else {"samples": torch.zeros((1, 4, 64, 64))}
                dummy_image = image if is_image_input else torch.zeros((1, 64, 64, 3))
                return (dummy_latent, dummy_image)

            depth_cache_valid = (
                self.cached_input_hash_depth == current_input_hash and
                self.cached_depth_map is not None and
                self.cached_depth_model_name == ai_depth_model
            )

            if is_depth_needed and not depth_cache_valid:
                self._run_and_cache_analysis(decoded_image, True, depth_model_name=ai_depth_model)
                self.cached_input_hash_depth = current_input_hash
                self.cached_depth_model_name = ai_depth_model

            image_to_process = decoded_image

            if enable_auto_tone:
                tone_analysis = self._perform_unified_tone_analysis(image_to_process)
            else:
                tone_analysis = None

            image_bchw = image_to_process.permute(0, 3, 1, 2)
            
            if is_dof_active and self.cached_depth_map is not None:
                image_bchw = self._apply_dof_depth_only(image_bchw, self.cached_depth_map, ai_dof_strength, ai_dof_focus_depth)
            
            final_bhwc = image_bchw.permute(0, 2, 3, 1)

            if enable_auto_tone or ai_colors or ai_details or ai_relight:
                final_bhwc = self._apply_unified_color_and_tone_pipeline(
                    final_bhwc,
                    ai_colors, ai_colors_strength,
                    ai_details, ai_details_strength,
                    ai_relight, ai_relight_strength,
                    enable_auto_tone, tone_analysis, auto_tone_strength
                )

            final_bchw = final_bhwc.permute(0, 3, 1, 2)
            corrected_bchw = self.apply_final_clipping_protection(final_bchw)
            final_image = corrected_bchw.permute(0, 2, 3, 1)

            if enable_upscale and upscale_factor > 1.0:
                h_orig, w_orig = final_image.shape[1], final_image.shape[2]
                target_h, target_w = int(h_orig * upscale_factor), int(w_orig * upscale_factor)
                if upscale_model == "Simple: Bicubic (Standard)":
                    final_image = F.interpolate(final_image.movedim(-1, 1), size=(target_h, target_w), mode='bicubic', align_corners=False, antialias=True).movedim(1, -1)
                else:
                    loaded_model = self.load_upscaler_model(upscale_model)
                    if loaded_model:
                        ai_upscaled_image = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']().upscale(upscale_model=loaded_model, image=final_image)[0]
                        final_image = F.interpolate(ai_upscaled_image.movedim(-1, 1), size=(target_h, target_w), mode='bicubic', align_corners=False, antialias=True).movedim(1, -1)
                    else:
                        print(f"Warning: Upscaler model {upscale_model} failed to load. Skipping upscale.")

            final_image = final_image.to(device)
            
            refined_latent = None
            if vae is not None:
                clamped_final_image = torch.clamp(final_image, 0.0, 1.0)
                if use_tiled_vae:
                    encode_node = nodes.NODE_CLASS_MAPPINGS['VAEEncodeTiled']()
                    refined_latent = encode_node.encode(vae, clamped_final_image, tile_size, overlap=64)[0]
                else:
                    encode_node = nodes.NODE_CLASS_MAPPINGS['VAEEncode']()
                    refined_latent = encode_node.encode(vae, clamped_final_image)[0]
                if not isinstance(refined_latent, dict): refined_latent = {"samples": refined_latent}
            
            if refined_latent is None:
                h_final, w_final = final_image.shape[1], final_image.shape[2]
                refined_latent = {"samples": torch.zeros((1, 4, h_final // 8, w_final // 8), dtype=torch.float32, device=device)}

            return (refined_latent, final_image.cpu())

        except model_management.InterruptProcessingException:
            print("Refiner cancelled by user")
            self._invalidate_cache()
            raise
        except Exception as e:
            print(f"FATAL Error in Refiner: {e}")
            self._invalidate_cache()
            
            dummy_latent = {"samples": torch.zeros((1, 4, 64, 64))} if latent is None else latent
            dummy_image = torch.zeros((1, 64, 64, 3)) if image is None else image
            return (dummy_latent, dummy_image)
        
    def _apply_unified_color_and_tone_pipeline(self, image_tensor_bhwc,
                                                ai_colors, ai_colors_strength,
                                                ai_details, ai_details_strength,
                                                ai_relight, ai_relight_strength,
                                                enable_auto_tone, tone_analysis, auto_tone_strength):
        try:
            check_for_interruption()
            
            if not ai_colors and not ai_details and not enable_auto_tone and not ai_relight:
                return image_tensor_bhwc

            image_bchw = image_tensor_bhwc.permute(0, 3, 1, 2)
            
            lab_bchw = kornia.color.rgb_to_lab(image_bchw)
            l_channel, a_channel, b_channel = lab_bchw[:, 0:1], lab_bchw[:, 1:2], lab_bchw[:, 2:3]

            relight_boost_map = None
            if ai_relight and ai_relight_strength > 0:
                l_channel, a_channel, b_channel, relight_boost_map = self.apply_ai_relight(l_channel, a_channel, b_channel, ai_relight_strength)

            if ai_colors and ai_colors_strength > 0.0:
                l_channel, a_channel, b_channel = self._apply_intelligent_vibrance(
                    l_channel, a_channel, b_channel, ai_colors_strength, image_tensor_bhwc.device, relight_boost_map
                )

            if ai_details and ai_details_strength > 0.0:
                l_normalized_details = l_channel / 100.0
                details_enhancement = self.apply_ai_details(l_normalized_details, ai_details_strength)
                l_channel = torch.clamp(details_enhancement, 0.0, 1.0) * 100.0

            if enable_auto_tone and tone_analysis is not None:
                l_normalized_tone = l_channel / 100.0
                l_exposed = self._apply_auto_exposure(
                    l_normalized_tone,
                    tone_analysis["black_deepening_reqs"],
                    auto_tone_strength=auto_tone_strength
                )
                corrected_l_normalized = self._intelligently_deepen_blacks(
                    l_exposed, a_channel, b_channel, 0.02,
                    black_deepening_reqs=tone_analysis["black_deepening_reqs"],
                    auto_tone_strength=auto_tone_strength
                )
                l_channel = corrected_l_normalized * 100.0


            final_lab = torch.cat([l_channel, a_channel, b_channel], dim=1)
            final_rgb_bchw = kornia.color.lab_to_rgb(final_lab)
            
            return final_rgb_bchw.permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL Error during unified color/tone pipeline: {e}")
            return image_tensor_bhwc

    def detect_clipping_issues(self, rgb_bchw):
        try:
            check_for_interruption()
            
            TRUE_BLACK = 0.001
            TRUE_WHITE = 0.999
            NEAR_BLACK_THRESHOLD = 0.02
            NEAR_WHITE_THRESHOLD = 0.985
            
            r_channel = rgb_bchw[:, 0:1]
            g_channel = rgb_bchw[:, 1:2] 
            b_channel = rgb_bchw[:, 2:3]
            
            true_highlight_clipping = torch.any(rgb_bchw >= TRUE_WHITE, dim=1, keepdim=True)
            true_shadow_clipping = torch.any(rgb_bchw <= TRUE_BLACK, dim=1, keepdim=True)
            near_highlight_stress = torch.any(rgb_bchw >= NEAR_WHITE_THRESHOLD, dim=1, keepdim=True)
            near_shadow_stress = torch.any(rgb_bchw <= NEAR_BLACK_THRESHOLD, dim=1, keepdim=True)
            
            individual_channel_clipping = (
                (r_channel >= TRUE_WHITE) | (r_channel <= TRUE_BLACK) |
                (g_channel >= TRUE_WHITE) | (g_channel <= TRUE_BLACK) |
                (b_channel >= TRUE_WHITE) | (b_channel <= TRUE_BLACK)
            )
            
            luminance = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
            luminance_clipping = (luminance >= TRUE_WHITE) | (luminance <= TRUE_BLACK)
            
            color_only_clipping = individual_channel_clipping & (~luminance_clipping)
            
            return {
                'true_highlight_clipping': true_highlight_clipping,
                'true_shadow_clipping': true_shadow_clipping,
                'near_highlight_stress': near_highlight_stress,
                'near_shadow_stress': near_shadow_stress,
                'color_only_clipping': color_only_clipping,
                'luminance_clipping': luminance_clipping,
                'needs_any_correction': (true_highlight_clipping | true_shadow_clipping | 
                                    near_highlight_stress | near_shadow_stress | color_only_clipping),
                'luminance': luminance
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return {
                'true_highlight_clipping': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'true_shadow_clipping': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'near_highlight_stress': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'near_shadow_stress': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'color_only_clipping': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'luminance_clipping': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'needs_any_correction': torch.zeros_like(rgb_bchw[:, 0:1], dtype=torch.bool),
                'luminance': 0.299 * rgb_bchw[:, 0:1] + 0.587 * rgb_bchw[:, 1:2] + 0.114 * rgb_bchw[:, 2:3]
            }
    def apply_ai_relight(self, l_channel, a_channel, b_channel, strength):
        try:
            check_for_interruption()
            if strength <= 0 or self.cached_depth_map is None:
                return l_channel, a_channel, b_channel, None
            
            device = l_channel.device
            l_normalized = l_channel / 100.0
            
            depth_map = self.cached_depth_map.to(device)
            if depth_map.shape[-2:] != l_normalized.shape[-2:]:
                depth_map = F.interpolate(depth_map, size=l_normalized.shape[-2:], mode="bilinear", align_corners=False)
            
            subject_mask = torch.pow(depth_map, 1.2)
            subject_mask = kornia.filters.gaussian_blur2d(subject_mask, kernel_size=(21, 21), sigma=(5.0, 5.0))
            
            relight_curve = 3.0 * l_normalized * torch.pow(1.0 - l_normalized, 2.0)
            boost_map = relight_curve * strength * subject_mask
            
            enhanced_l = l_normalized + boost_map
            
            original_chroma = torch.sqrt(a_channel**2 + b_channel**2 + 1e-6)
            luminance_ratio = enhanced_l / (l_normalized + 1e-6)
            chroma_boost_factor = 1.0 + (luminance_ratio - 1.0) * 0.7 * subject_mask
            
            enhanced_a = a_channel * chroma_boost_factor
            enhanced_b = b_channel * chroma_boost_factor
            
            new_chroma = torch.sqrt(enhanced_a**2 + enhanced_b**2 + 1e-6)
            max_chroma = 128.0
            chroma_scale = torch.where(new_chroma > max_chroma, max_chroma / new_chroma, torch.ones_like(new_chroma))
            
            enhanced_a = enhanced_a * chroma_scale
            enhanced_b = enhanced_b * chroma_scale
            
            print(f"[Relight] Strength: {strength}")
            print(f"[Relight] L: {l_normalized.mean():.3f} -> {enhanced_l.mean():.3f}")
            print(f"[Relight] Chroma: {original_chroma.mean():.1f} -> {(torch.sqrt(enhanced_a**2 + enhanced_b**2)).mean():.1f}")
            
            return (enhanced_l.clamp(0.0, 1.0) * 100.0), enhanced_a, enhanced_b, boost_map
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Relight error: {e}")
            return l_channel, a_channel, b_channel, None
    def apply_camera_raw_style_tone_mapping(self, rgb_bchw, clipping_analysis):
        try:
            check_for_interruption()
            
            lab_bchw = kornia.color.rgb_to_lab(rgb_bchw)
            l_channel, a_channel, b_channel = lab_bchw[:, 0:1], lab_bchw[:, 1:2], lab_bchw[:, 2:3]
            l_normalized = l_channel / 100.0
            
            corrected_l_normalized = l_normalized.clone()

            if torch.any(clipping_analysis['true_highlight_clipping'] | clipping_analysis['near_highlight_stress']):
                target_ceiling = 0.999
                protection_threshold = 0.98
                high_end_mask = corrected_l_normalized > protection_threshold
                
                if torch.any(high_end_mask):
                    rel_pos = (corrected_l_normalized - protection_threshold) / (1.0 - protection_threshold)
                    compressed_highs = protection_threshold + (rel_pos * (target_ceiling - protection_threshold))
                    corrected_l_normalized = torch.where(high_end_mask, compressed_highs, corrected_l_normalized)
            
            if torch.any(clipping_analysis['true_shadow_clipping'] | clipping_analysis['near_shadow_stress']):
                target_floor = 0.001
                protection_threshold = 0.02 
                low_end_mask = corrected_l_normalized < protection_threshold
                
                if torch.any(low_end_mask):
                    rel_pos = corrected_l_normalized / protection_threshold
                    compressed_lows = (rel_pos * (protection_threshold - target_floor)) + target_floor
                    corrected_l_normalized = torch.where(low_end_mask, compressed_lows, corrected_l_normalized)

            final_l_channel = torch.clamp(corrected_l_normalized * 100.0, 0.0, 100.0)
            final_lab = torch.cat([final_l_channel, a_channel, b_channel], dim=1)

            result_rgb = kornia.color.lab_to_rgb(final_lab)
            return torch.clamp(result_rgb, 0, 1)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return rgb_bchw
    
    def _apply_dof_depth_only(self, image_bchw, depth_map, dof_strength, ai_dof_focus_depth):

        try:
            check_for_interruption()

            if depth_map is None:
                print("Refiner: DOF skipped, missing depth map.")
                return image_bchw

            h, w = image_bchw.shape[-2:]
            if depth_map.shape[-2:] != (h, w):
                depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)

            calibrated_strength = dof_strength ** 1.5
            scale_factor = min(h, w) / 512.0
            
            foreground_threshold = torch.quantile(depth_map, 0.90)
            foreground_mask = depth_map > foreground_threshold
            
            foreground_depths = depth_map[foreground_mask]
            
            if foreground_depths.numel() > 0:
                focus_plane = torch.mean(foreground_depths)
            else:
                focus_plane = torch.quantile(depth_map, 0.96)
            
            focus_tolerance = 0.06 * (1.2 - calibrated_strength) + (1.0 - ai_dof_focus_depth) * 1.0
            distance_from_focus = torch.abs(depth_map - focus_plane)
            
            intensity_map = torch.clamp(distance_from_focus - focus_tolerance, 0.0, 1.0)
            map_max = torch.max(intensity_map)
            if map_max > 0:
                intensity_map = intensity_map / map_max
            
            map_h, map_w = intensity_map.shape[-2:]
            downsampled_map = F.interpolate(intensity_map, size=(map_h // 8, map_w // 8), mode='bilinear', align_corners=False)
            
            smoothing_radius = max(2.0, 4.0 * (scale_factor / 8.0))
            smoothing_kernel = int(smoothing_radius * 2) | 1
            blurred_downsampled_map = kornia.filters.gaussian_blur2d(downsampled_map, (smoothing_kernel, smoothing_kernel), (smoothing_radius, smoothing_radius))
            
            smooth_intensity_map = F.interpolate(blurred_downsampled_map, size=(map_h, map_w), mode='bilinear', align_corners=False)
            smooth_intensity_map = torch.pow(smooth_intensity_map, 1.2)

            max_blur_radius = calibrated_strength * 12.0 * scale_factor
            
            blur_ratios = [0.15, 0.4, 1.0]
            blurred_images = [image_bchw]

            for ratio in blur_ratios:
                radius = max_blur_radius * ratio
                if radius >= 1.0:
                    kernel_size = int(radius * 2) | 1
                    blurred_images.append(kornia.filters.gaussian_blur2d(image_bchw, (kernel_size, kernel_size), (radius, radius)))
                else:
                    blurred_images.append(image_bchw)

            final_image = blurred_images[0]
            zone_boundaries = [0.0, 0.33, 0.66, 1.0]

            for i in range(len(blur_ratios)):
                start_boundary = zone_boundaries[i]
                end_boundary = zone_boundaries[i+1]
                
                alpha = (smooth_intensity_map - start_boundary) / (end_boundary - start_boundary + 1e-7)
                alpha = torch.clamp(alpha, 0.0, 1.0)
                
                final_image = torch.lerp(final_image, blurred_images[i+1], alpha)

            return final_image

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"FATAL: Error in _apply_dof_depth_only: {e}")
            return image_bchw
        
    def apply_intelligent_desaturation_with_lab(self, rgb_bchw, clipping_analysis, lab_bchw):
        try:
            check_for_interruption()
            
            color_clipping_mask = clipping_analysis.get('color_only_clipping')
            if color_clipping_mask is None or not torch.any(color_clipping_mask):
                return lab_bchw

            l_channel, a_channel, b_channel = lab_bchw[:, 0:1], lab_bchw[:, 1:2], lab_bchw[:, 2:3]
            
            h, w = color_clipping_mask.shape[-2:]
            blur_radius = max(2.0, min(h, w) * 0.003)
            kernel_size = min(int(blur_radius * 2) | 1, 9)
            smooth_clipping_mask = kornia.filters.gaussian_blur2d(
                color_clipping_mask.float(), (kernel_size, kernel_size), (blur_radius, blur_radius)
            )

            working_mask = smooth_clipping_mask > 0.05
            if not torch.any(working_mask):
                return lab_bchw

            low_desat = torch.zeros_like(a_channel)
            high_desat = torch.ones_like(a_channel)

            for _ in range(4):
                mid_desat = (low_desat + high_desat) * 0.5
                test_a = a_channel * (1.0 - mid_desat)
                test_b = b_channel * (1.0 - mid_desat)
                test_lab = torch.cat([l_channel, test_a, test_b], dim=1)
                test_rgb = kornia.color.lab_to_rgb(test_lab)
                is_safe = torch.all((test_rgb >= -0.01) & (test_rgb <= 1.01), dim=1, keepdim=True)
                low_desat = torch.where(is_safe, low_desat, mid_desat)
                high_desat = torch.where(is_safe, mid_desat, high_desat)
            
            final_desat_amount = low_desat * smooth_clipping_mask
            corrected_a = a_channel * (1.0 - final_desat_amount)
            corrected_b = b_channel * (1.0 - final_desat_amount)
            
            return torch.cat([l_channel, corrected_a, corrected_b], dim=1)
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return lab_bchw
   
    def _reconstruct_clipped_highlights(self, rgb_bchw, clipping_analysis):
        try:
            check_for_interruption()
            reconstruction_mask = clipping_analysis['color_only_clipping']
            if not torch.any(reconstruction_mask):
                return rgb_bchw
            
            result = rgb_bchw.clone()
            safe_rgb = torch.where(rgb_bchw >= 1.0, torch.tensor(-1.0).to(rgb_bchw.device), rgb_bchw)
            max_unclipped_value, _ = torch.max(safe_rgb, dim=1, keepdim=True)
            max_unclipped_value = torch.clamp(max_unclipped_value, 0.0, 1.0)
            
            original_ratios = rgb_bchw / (max_unclipped_value + 1e-6)
            new_max_value = max_unclipped_value * 0.98
            reconstructed_rgb = original_ratios * new_max_value
            
            h, w = reconstruction_mask.shape[-2:]
            blur_radius = max(1.0, min(h, w) * 0.002)
            kernel_size = int(blur_radius * 2) | 1
            smooth_reconstruction_mask = kornia.filters.gaussian_blur2d(
                reconstruction_mask.float(), (kernel_size, kernel_size), (blur_radius, blur_radius)
            )
            final_result = torch.lerp(result, reconstructed_rgb, smooth_reconstruction_mask)
            return torch.clamp(final_result, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return rgb_bchw
    
    def apply_final_clipping_protection(self, rgb_bchw):
        try:
            check_for_interruption()
            
            initial_analysis = self.detect_clipping_issues(rgb_bchw)
            if not torch.any(initial_analysis['needs_any_correction']):
                return rgb_bchw

            tone_mapped_rgb = self.apply_camera_raw_style_tone_mapping(rgb_bchw, initial_analysis)

            reconstruction_analysis = self.detect_clipping_issues(tone_mapped_rgb)
            if torch.any(reconstruction_analysis['color_only_clipping']):
                reconstructed_rgb = self._reconstruct_clipped_highlights(tone_mapped_rgb, reconstruction_analysis)
            else:
                reconstructed_rgb = tone_mapped_rgb

            final_analysis = self.detect_clipping_issues(reconstructed_rgb)
            if torch.any(final_analysis['color_only_clipping']):
                lab_for_desat = kornia.color.rgb_to_lab(reconstructed_rgb)
                corrected_lab_desat = self.apply_intelligent_desaturation_with_lab(reconstructed_rgb, final_analysis, lab_for_desat)
                final_rgb = kornia.color.lab_to_rgb(corrected_lab_desat)
            else:
                final_rgb = reconstructed_rgb

            clamped_rgb = torch.clamp(final_rgb, 0.0, 1.0)
            safe_rgb = self._enforce_8bit_safety(clamped_rgb)
            
            return safe_rgb
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return torch.clamp(rgb_bchw, 0, 1)

    def _enforce_8bit_safety(self, rgb_bchw):
        safe_floor = 0.004
        safe_ceiling = 0.996
        low_threshold = 0.02
        high_threshold = 0.98
        
        low_mask = rgb_bchw < low_threshold
        if torch.any(low_mask):
            rel_pos = rgb_bchw / low_threshold
            compressed = safe_floor + (rel_pos * (low_threshold - safe_floor))
            rgb_bchw = torch.where(low_mask, compressed, rgb_bchw)

        high_mask = rgb_bchw > high_threshold
        if torch.any(high_mask):
            rel_pos = (rgb_bchw - high_threshold) / (1.0 - high_threshold)
            compressed = high_threshold + (rel_pos * (safe_ceiling - high_threshold))
            rgb_bchw = torch.where(high_mask, compressed, rgb_bchw)

        return rgb_bchw    

    def _intelligently_deepen_blacks(self, luminance, lab_a, lab_b, safe_shadow_level,
                                    black_deepening_reqs=None, auto_tone_strength=1.0):
        
        if auto_tone_strength <= 0.0 or black_deepening_reqs is None:
            return luminance

        blacks_base = black_deepening_reqs.get('recommended_blacks', -30.0)
        shadows_base = black_deepening_reqs.get('recommended_shadows', 10.0)
        contrast_base = black_deepening_reqs.get('recommended_contrast', 10.0)
        

        s = float(auto_tone_strength)

        mB_half, mB_full = 2.0, 2.5
        mC_half, mC_full = 2.0, 3.0

        if s <= 0.0:
            blacks_amount = 0.0
            shadows_amount = 0.0
            contrast_amount = 0.0

        elif s <= 0.5:
            t = s / 0.5
            t2 = t * t
            blacks_amount   = blacks_base   * (mB_half * t2)
            contrast_amount = contrast_base * (mC_half * t2)
            shadows_amount  = shadows_base  * t2

        else:
            t = (s - 0.5) / 0.5
            t2 = t * t
            blacks_amount   = blacks_base   * (mB_half + (mB_full - mB_half) * t2)
            contrast_amount = contrast_base * (mC_half + (mC_full - mC_half) * t2)
            shadows_amount  = shadows_base

        
        p01 = black_deepening_reqs.get('p01', 0.0)
        p05 = black_deepening_reqs.get('p05', 0.05)
        
        result = luminance.clone()
        
        blacks_range_end = p01 + 0.15
        blacks_mask = torch.clamp((blacks_range_end - luminance) / (blacks_range_end - p01 + 1e-6), 0.0, 1.0)
        blacks_mask = blacks_mask.pow(1.8)
        
        blacks_adjustment = blacks_amount / 100.0
        gamma = 1.0 + abs(blacks_adjustment) * 1.2
        darkened = torch.pow(result + 1e-6, gamma)

        strength = torch.clamp(blacks_mask * (0.4 + 0.8 * abs(blacks_adjustment)), 0.0, 1.0)
        result = torch.lerp(result, darkened, strength)

        
        shadows_center = p05 + 0.15
        shadows_mask = torch.exp(-((luminance - shadows_center) ** 2) / (2 * 0.15 ** 2))
        
        if shadows_amount > 0:
            shadow_gamma = 1.0 - (shadows_amount / 100.0) * 0.3
            lifted = torch.pow(result + 1e-6, shadow_gamma)
            result = torch.lerp(result, lifted, shadows_mask)
        elif shadows_amount < 0:
            shadow_gamma = 1.0 + (abs(shadows_amount) / 100.0) * 0.3
            lowered = torch.pow(result + 1e-6, shadow_gamma)
            result = torch.lerp(result, lowered, shadows_mask)
        
        if abs(contrast_amount) > 1.0:
            midtone_mask = torch.exp(-((luminance - 0.5) ** 2) / (2 * 0.2 ** 2))
            contrast_factor = 1.0 + (contrast_amount / 100.0) * 3.0
            deviation = result - 0.5
            result = 0.5 + deviation * (1.0 + (contrast_factor - 1.0) * midtone_mask)
        
        chroma = torch.sqrt(lab_a ** 2 + lab_b ** 2 + 1e-6)
        sat_protect = torch.clamp(1.0 - (chroma / 90.0), 0.5, 1.0)
        result = torch.lerp(luminance, result, sat_protect)
        
        return torch.clamp(result, 0.0, 1.0)
    
    def _apply_auto_exposure(self, luminance, black_deepening_reqs, auto_tone_strength=1.0):

        try:
            if auto_tone_strength <= 0.0 or black_deepening_reqs is None:
                return luminance

            base_ev = black_deepening_reqs.get("recommended_exposure_ev", 0.0)
            s = float(auto_tone_strength)
            s = max(0.0, min(1.0, s))

            if s <= 0.6:
                t = s / 0.6
                scale = t * t
            else:
                scale = 1.0

            ev = base_ev * scale

            if ev > 0:
                ev *= 0.34
            else:
                ev *= 0.2
                ev = max(ev, -0.08)

            if abs(ev) < 0.02:
                return luminance

            factor = float(2.0 ** ev)
            
            if factor < 1.0:
                result = torch.clamp(luminance * factor, 0.0, 1.0)
                return result
            
            
            p95 = black_deepening_reqs.get('p95', 0.90)
            
            transition_start = 0.50
            transition_end = torch.clamp(p95, 0.85, 0.98)
            
            t = torch.clamp((luminance - transition_start) / (transition_end - transition_start + 1e-6), 0.0, 1.0)
            t_smooth = t * t * (3.0 - 2.0 * t)
            
            adaptive_factor = factor * (1.0 - t_smooth) + 1.0 * t_smooth
            
            result = luminance * adaptive_factor
            
            return result.clamp(0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception:
            return luminance

    def _analyze_black_deepening_requirements(self, luminance):
        try:
            check_for_interruption()
            
            flattened = luminance.flatten()
            n = flattened.numel()
            
            if n > 100000:
                sample_indices = torch.randperm(n, device=luminance.device)[:50000]
                sample = flattened[sample_indices]
            else:
                sample = flattened
            
            lum_mean = torch.mean(sample)
            lum_std = torch.std(sample)
            
            p01 = torch.quantile(sample, 0.01)
            p05 = torch.quantile(sample, 0.05)
            p95 = torch.quantile(sample, 0.95)
            
            shadows = torch.mean((sample < 0.25).float())
            midtones = torch.mean(((sample >= 0.30) & (sample <= 0.70)).float())
            highlights = torch.mean((sample > 0.75).float())
            
            brightness_factor = torch.clamp((lum_mean - 0.35) / 0.30, 0.0, 1.0)
            
            is_high_key_artistic = (lum_mean > 0.55 and shadows < 0.10 and midtones > 0.65 and lum_std > 0.08 and lum_std < 0.18)
            is_low_key_artistic = (lum_mean < 0.40 and highlights < 0.15 and shadows > 0.30)
            is_bimodal = (shadows > 0.20 and highlights > 0.20)
            is_likely_stylized = is_high_key_artistic or is_low_key_artistic or is_bimodal
            
            recommended_blacks = -27.0 - (p01 * 145.0)
            recommended_blacks = float(torch.clamp(recommended_blacks, -85.0, -20.0))
            
            recommended_shadows = brightness_factor * 23.0 - 5.0
            
            contrast_ratio = lum_std / (lum_mean.clamp(min=0.3))
            contrast_need = torch.clamp((0.35 - contrast_ratio) / 0.20, 0.0, 1.0)
            recommended_contrast = 6.0 + (contrast_need * 10.0)
            
            if is_likely_stylized:
                recommended_blacks *= 0.75
                recommended_shadows *= 0.85
                recommended_contrast *= 0.8
            

            brightness_deviation = (0.50 - lum_mean) / 0.20

            brightness_deviation = torch.clamp(brightness_deviation, -1.5, 1.5)

            headroom = torch.clamp((0.97 - p95) / 0.20, 0.0, 1.0)

            if brightness_deviation > 0:
                exposure_ev = brightness_deviation * headroom
            else:
                exposure_ev = torch.clamp(brightness_deviation, -0.5, 0.0)

            if is_likely_stylized:
                exposure_ev = exposure_ev * 0.6

            recommended_exposure_ev = float(exposure_ev)


            return {
                'recommended_blacks': float(recommended_blacks),
                'recommended_shadows': float(recommended_shadows),
                'recommended_contrast': float(recommended_contrast),
                'recommended_exposure_ev': recommended_exposure_ev,
                'is_likely_stylized': bool(is_likely_stylized),
                'brightness_factor': float(brightness_factor),
                'p01': float(p01),
                'p05': float(p05),
                'p95': float(p95),
            }

            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return {'recommended_blacks': -30.0, 'recommended_shadows': 10.0, 'recommended_contrast': 10.0, 'is_likely_stylized': False, 'brightness_factor': 0.5, 'p01': 0.0, 'p05': 0.0, 'p95': 1.0}
        
    def _perform_unified_tone_analysis(self, image_tensor_bhwc):

        try:
            image_bchw = image_tensor_bhwc.permute(0, 3, 1, 2)
            lab_bchw = kornia.color.rgb_to_lab(image_bchw)
            l_channel = lab_bchw[:, 0:1]
            l_normalized = l_channel / 100.0

            black_deepening_reqs = self._analyze_black_deepening_requirements(l_normalized)

            analysis_results = {
                "black_deepening_reqs": black_deepening_reqs,
            }
            return analysis_results

        except Exception as e:
            print(f"FATAL: Error in unified tone analysis: {e}")
            return None

    def _optimized_gaussian_blur(self, tensor, radius, max_kernel_size=15):
        try:
            check_for_interruption()
            kernel_size = max(3, int(radius * 2) | 1)
            kernel_size = min(kernel_size, max_kernel_size)
            if tensor.dim() == 4:
                return kornia.filters.gaussian_blur2d(tensor, (kernel_size, kernel_size), (radius, radius))
            else:
                tensor_4d = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor.unsqueeze(0).unsqueeze(0)
                blurred = kornia.filters.gaussian_blur2d(tensor_4d, (kernel_size, kernel_size), (radius, radius))
                return blurred.squeeze() if tensor.dim() < 4 else blurred
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return tensor

    def _simple_neighbor_smooth(self, luminance):
        try:
            padded = F.pad(luminance, (1, 1, 1, 1), mode='reflect')
            avg_neighbors = F.avg_pool2d(padded, kernel_size=3, stride=1, padding=0) * (9/8) - luminance / 8
            return luminance * 0.6 + avg_neighbors * 0.4
        except Exception as e:
            return luminance

    def _apply_intelligent_vibrance(self, l_channel, a_channel, b_channel, strength, device, relight_boost_map=None):
    
        if strength <= 0:
            return l_channel, a_channel, b_channel

        chroma = torch.sqrt(a_channel**2 + b_channel**2) + 1e-6
        l_normalized = l_channel / 100.0

        median_chroma = torch.median(chroma)
        mean_luminance = torch.mean(l_normalized)
        chroma_p75 = torch.quantile(chroma, 0.75)
        
        is_singular_tone = (median_chroma > 20.0) and (mean_luminance > 0.55)
        
        if is_singular_tone:
            context_scale = 0.70
        else:
            context_scale = 1.15
        
        effective_strength = strength * 2.8 * context_scale
        
        if is_singular_tone:
            adaptive_ceiling = torch.clamp(chroma_p75 * 1.5, 30.0, 65.0)
        else:
            adaptive_ceiling = torch.clamp(chroma_p75 * 1.8, 35.0, 80.0)

        vibrance_mask = torch.clamp(1.0 - (chroma / adaptive_ceiling), 0.05, 1.0)
        vibrance_mask = vibrance_mask ** 2.0
        
        skin_tone_range = torch.exp(-((l_normalized - 0.65) ** 2) / (2 * 0.20 ** 2))
        is_subtle_color = (chroma < 25.0).float()
        base_protection = skin_tone_range * is_subtle_color
        
        strength_adjustment = torch.ones_like(vibrance_mask)
        if relight_boost_map is not None:
            normalized_boost = relight_boost_map / (relight_boost_map.max() + 1e-6)
            
            strength_adjustment = 1.0 - (normalized_boost * 0.6)
        
        final_protection = torch.maximum(base_protection * 0.7, (1.0 - strength_adjustment))
        final_vibrance_mask = vibrance_mask * (1.0 - final_protection)

        boost_scalar = effective_strength * final_vibrance_mask
        scale_factor = 1.0 + boost_scalar

        new_chroma = chroma * scale_factor
        
        max_chroma = 135.0
        new_chroma_limited = max_chroma * torch.tanh(new_chroma / max_chroma)

        scale_ratio = new_chroma_limited / chroma
        new_a = a_channel * scale_ratio
        new_b = b_channel * scale_ratio

        return l_channel, new_a, new_b
        
    def create_optimized_edge_mask(self, luminance, device, dtype):

        try:
            check_for_interruption()
            edges = kornia.filters.sobel(luminance)
            if edges.dim() == 4:
                edge_mag = edges.abs().mean(1, keepdim=True)
            else:
                edge_mag = edges.abs()

            edge_mask = torch.tanh(edge_mag * 6.0)
            edge_mask = torch.clamp(edge_mask, 0.0, 1.0)
            return edge_mask
        except model_management.InterruptProcessingException:
            raise
        except Exception:
            return torch.zeros_like(luminance)


    def advanced_gaussian_blur(self, tensor, radius, device, dtype):
        try:
            check_for_interruption()
            kernel_size = max(3, int(radius * 2) | 1)
            kernel_size = min(kernel_size, 31)
            sigma = radius / 2.0
            tensor_bchw = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
            blurred = kornia.filters.gaussian_blur2d(
                tensor_bchw, 
                kernel_size=(kernel_size, kernel_size), 
                sigma=(sigma, sigma)
            )
            return blurred if tensor.dim() == 4 else blurred.squeeze(0)
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            return tensor
    
    def apply_ai_details(self, luminance, strength):

        try:
            check_for_interruption()
            if strength <= 0:
                return luminance

            device = luminance.device
            dtype = luminance.dtype

            s = float(max(0.0, min(2.0, strength)))
            base_strength = (min(s, 1.0) * 0.7) + max(s - 1.0, 0.0) * 0.3
            if base_strength <= 0:
                return luminance

            h = luminance.shape[-2]
            w = luminance.shape[-1]
            scale = min(h, w) / 1024.0
            base_radius = 2.5 * scale
            base_radius = max(1.5, min(3.5, base_radius))

            r_fine = max(1.0, base_radius * 0.5)
            r_mid = base_radius
            r_coarse = base_radius * 1.5

            blur_fine = self._optimized_gaussian_blur(luminance, radius=r_fine)
            blur_mid = self._optimized_gaussian_blur(luminance, radius=r_mid)
            blur_coarse = self._optimized_gaussian_blur(luminance, radius=r_coarse)

            detail_fine = luminance - blur_fine
            detail_mid = luminance - blur_mid
            detail_coarse = luminance - blur_coarse

            noise_gate = 0.025

            def _gate(d):
                ad = d.abs()
                mask = (ad > noise_gate).to(d.dtype)
                return d * mask

            detail_fine = _gate(detail_fine)
            detail_mid = _gate(detail_mid)
            detail_coarse = _gate(detail_coarse)

            detail = (0.4 * detail_mid) + (0.35 * detail_fine) + (0.25 * detail_coarse)

            try:
                edge_mask = self.create_optimized_edge_mask(luminance, device, dtype)
                safe_edge_mask = 1.0 - 0.6 * edge_mask
                safe_edge_mask = torch.clamp(safe_edge_mask, 0.4, 1.0)
                detail = detail * safe_edge_mask
            except Exception:
                pass

            enhanced_detail = detail * base_strength

            limit = 0.08
            enhanced_detail = limit * torch.tanh(enhanced_detail / limit)

            if getattr(self, "cached_depth_map", None) is not None:
                d_map = self.cached_depth_map.to(device=device, dtype=dtype)
                if d_map.dim() == 3:
                    d_map = d_map.unsqueeze(1)
                d_map = F.interpolate(
                    d_map,
                    size=luminance.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                depth_mask = 0.25 + (0.75 * d_map)
                enhanced_detail = enhanced_detail * depth_mask

            result = luminance + enhanced_detail
            return torch.clamp(result, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Details error: {e}")
            return luminance

    