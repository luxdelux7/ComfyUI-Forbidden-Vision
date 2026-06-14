import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import nodes
import kornia
import math
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

                "neural_corrector": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "corrector_tone": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "corrector_color": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),                
                
                "depth_relight": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "depth_relight_stringth": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),

                "depth_dof_enable": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "depth_dof_strength": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}), 
                "depth_dof_focus": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.99, "step": 0.01}),

                "depth_dof_model": (["V2-Small", "V2-Base"], {"default": "V2-Small"}),
                
                "maintain_aspect_ratio": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "If Enabled: Crops edges slightly to fit grid without stretching."}),
                "enforce_mod32_boundaries": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Forces output resolution to be a multiple of 32."}),

                
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "offload_models_to_cpu": ("BOOLEAN", {"default": True, "tooltip": "Move neural corrector and depth models from VRAM to RAM after processing. Frees GPU memory for other nodes (e.g. checkpoint swaps in queued workflows). Disable if you're chaining many refiner passes and want to skip the small reload overhead."}),
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
        self.cached_depth_map = None
        self.cached_decoded_image = None
        self.cached_vae_hash = None
        self.cached_input_hash = None
        self.cached_input_hash_depth = None
        self.cached_depth_model_name = None
        self.cached_depth_settings = None

    def _get_vae_hash(self, vae):
        if vae is None: return None
        try: return hash((id(vae), str(vae.device) if hasattr(vae, 'device') else 'unknown'))
        except Exception: return id(vae)

    def _get_tensor_hash(self, tensor):
        """
        Stable hash that doesn't depend on data_ptr() (which changes every allocation).
        Uses shape + dtype + a sampled content hash so it's cheap but reliable.
        """
        if tensor is None:
            return None
        try:
            t = tensor.detach().cpu()
            flat = t.flatten()
            n = flat.numel()
            if n == 0:
                return hash((tuple(tensor.shape), str(tensor.dtype)))
            step = max(1, n // 1024)
            sampled = flat[::step].float().numpy().tobytes()
            return hash((tuple(tensor.shape), str(tensor.dtype), sampled))
        except Exception:
            return hash((tuple(tensor.shape), str(tensor.dtype)))
    
    def _run_and_cache_analysis(self, image_tensor, run_depth, depth_model_name="V2-Small"):
        try:
            check_for_interruption()

            h, w = image_tensor.shape[1], image_tensor.shape[2]
            device = image_tensor.device
            dtype = image_tensor.dtype

            if run_depth:
                self.cached_depth_map = None

            img_np_uint8 = None

            if run_depth:
                if img_np_uint8 is None:
                    img_np_uint8 = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)

                depth_tensor = self.depth_manager.infer_depth_full(img_np_uint8, depth_model_name)

                if depth_tensor is not None:
                    if depth_tensor.shape[-2:] != (h, w):
                        depth_tensor = F.interpolate(
                            depth_tensor, size=(h, w), mode="bilinear", align_corners=False
                        )

                    self.cached_depth_map = depth_tensor.to(
                        device=device,
                        dtype=dtype,
                        non_blocking=True
                    )


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
        except Exception as e:
            print(f"Error loading upscaler model {clean_model_name_val}: {e}")
            return None

    def _calculate_best_fit_resolution(self, w_orig, h_orig, scale_factor, alignment=32):
        target_w = w_orig * scale_factor
        target_h = h_orig * scale_factor
        target_area = target_w * target_h
        original_aspect = w_orig / h_orig
        
        best_w, best_h = 0, 0
        min_error = float('inf')
        
        center_w = int(round(target_w / alignment) * alignment)
        search_range = 2 
        
        for i in range(-search_range, search_range + 1):
            w_candidate = center_w + (i * alignment)
            if w_candidate <= 0: continue
            
            h_ideal = w_candidate / original_aspect
            h_candidate = int(round(h_ideal / alignment) * alignment)
            if h_candidate <= 0: continue
            
            candidate_area = w_candidate * h_candidate
            scale_deviation = abs(candidate_area - target_area) / target_area
            candidate_aspect = w_candidate / h_candidate
            ar_error = abs(candidate_aspect - original_aspect)
            total_error = (scale_deviation * 50.0) + (ar_error * 20.0)
            
            if total_error < min_error:
                min_error = total_error
                best_w = w_candidate
                best_h = h_candidate
        return best_h, best_w



    
    def _smart_resize_and_crop(self, image_bchw, target_h, target_w):
        B, C, H, W = image_bchw.shape
        target_ratio = target_w / target_h
        current_ratio = W / H
        
        if current_ratio > target_ratio:
            scale_factor = target_h / H
            temp_h = target_h
            temp_w = int(W * scale_factor)
        else:
            scale_factor = target_w / W
            temp_w = target_w
            temp_h = int(H * scale_factor)
            
        resized = F.interpolate(image_bchw, size=(temp_h, temp_w), mode='bicubic', align_corners=False, antialias=True)
        
        if temp_w == target_w and temp_h == target_h:
            return resized
            
        start_x = (temp_w - target_w) // 2
        start_y = (temp_h - target_h) // 2
        
        cropped = resized[:, :, start_y:start_y+target_h, start_x:start_x+target_w]
        return cropped
    
    def refine_and_process(self,
                        neural_corrector, corrector_tone, corrector_color,
                        enable_upscale, upscale_model, upscale_factor,
                        depth_relight, depth_relight_stringth,
                        depth_dof_enable, depth_dof_strength, depth_dof_focus,
                        depth_dof_model,
                        maintain_aspect_ratio, enforce_mod32_boundaries,
                        use_tiled_vae, tile_size,
                        offload_models_to_cpu=True,
                        latent=None, vae=None, image=None, **kwargs):
        try:
            check_for_interruption()
            # --- Sanitize inputs (ComfyUI can corrupt widgets during bypass/mute) ---
            if depth_dof_model not in ("V2-Small", "V2-Base"):
                depth_dof_model = "V2-Small"
            upscale_factor = max(1.0, min(8.0, float(upscale_factor)))
            corrector_tone = max(0.0, min(1.0, float(corrector_tone)))
            corrector_color = max(0.0, min(1.0, float(corrector_color)))
            depth_relight_stringth = max(0.0, min(1.0, float(depth_relight_stringth)))
            depth_dof_strength = max(0.0, min(1.0, float(depth_dof_strength)))
            depth_dof_focus = max(0.50, min(0.99, float(depth_dof_focus)))
            tile_size = max(256, min(2048, int(tile_size)))
            enable_upscale = bool(enable_upscale)
            neural_corrector = bool(neural_corrector)
            depth_relight = bool(depth_relight)
            depth_dof_enable = bool(depth_dof_enable)
            maintain_aspect_ratio = bool(maintain_aspect_ratio)
            enforce_mod32_boundaries = bool(enforce_mod32_boundaries)
            use_tiled_vae = bool(use_tiled_vae)
            offload_models_to_cpu = bool(offload_models_to_cpu)
            # --- End sanitization ---
            device = model_management.get_torch_device()

            is_latent_input = latent is not None and "samples" in latent
            is_image_input = image is not None

            if not is_latent_input and not is_image_input:
                print("Warning: No valid inputs provided.")
                return ({"samples": torch.zeros((1, 4, 64, 64))}, torch.zeros((1, 64, 64, 3)))

            is_dof_active = depth_dof_enable
            is_depth_needed = is_dof_active or (depth_relight and depth_relight_stringth > 0)

            input_key_tensor = latent["samples"] if is_latent_input else image
            current_input_hash = self._get_tensor_hash(input_key_tensor)

            decoded_image = None

            if is_image_input:
                decoded_image = image.to(device)
                self.cached_decoded_image = decoded_image
                self.cached_input_hash = current_input_hash
                self.cached_vae_hash = None

            elif is_latent_input and vae is not None:
                current_vae_hash = self._get_vae_hash(vae)

                decode_cache_valid = (
                    self.cached_decoded_image is not None
                    and self.cached_vae_hash == current_vae_hash
                    and self.cached_input_hash == current_input_hash
                )

                if decode_cache_valid:
                    decoded_image = self.cached_decoded_image
                else:
                    from .utils import normalize_vae_decode_output
                    decoded_image = normalize_vae_decode_output(vae.decode(input_key_tensor.to(device)))
                    self.cached_decoded_image = decoded_image
                    self.cached_vae_hash = current_vae_hash
                    self.cached_input_hash = current_input_hash
                    self.cached_input_hash_depth = None

            if decoded_image is None:
                # Preserve the input latent's actual shape rather than forcing 4D
                if is_latent_input:
                    dummy_latent = latent
                else:
                    dummy_latent = {"samples": torch.zeros((1, 4, 64, 64))}
                dummy_image = image if is_image_input else torch.zeros((1, 64, 64, 3))
                return (dummy_latent, dummy_image)

            if is_depth_needed:
                current_depth_settings = (depth_dof_model,)

                depth_cache_valid = (
                    self.cached_depth_map is not None
                    and self.cached_input_hash_depth == current_input_hash
                    and self.cached_depth_settings == current_depth_settings
                )

                if not depth_cache_valid:
                    self._run_and_cache_analysis(
                        decoded_image, True, depth_model_name=depth_dof_model
                    )
                    self.cached_input_hash_depth = current_input_hash
                    self.cached_depth_model_name = depth_dof_model
                    self.cached_depth_settings = current_depth_settings

            elif not is_depth_needed:
                if self.cached_depth_map is not None:
                    self.cached_depth_map = None
                    self.cached_input_hash_depth = None
                    self.cached_depth_settings = None

            image_to_process = decoded_image

            if neural_corrector:
                image_to_process = self._apply_neural_correction(
                    image_to_process,
                    corrector_tone,
                    corrector_color,
                )

            image_bchw = image_to_process.permute(0, 3, 1, 2)

            if depth_relight and depth_relight_stringth > 0:
                image_bchw = self._apply_relight_bchw(image_bchw, depth_relight_stringth)

            if is_dof_active and self.cached_depth_map is not None:
                image_bchw = self._apply_dof_depth_only(
                    image_bchw, self.cached_depth_map, depth_dof_strength, depth_dof_focus
                )

            final_image_bhwc = image_bchw.permute(0, 2, 3, 1)

            final_bchw = final_image_bhwc.permute(0, 3, 1, 2)
            final_bchw = self.apply_final_clipping_protection(final_bchw)
            final_image_bhwc = final_bchw.permute(0, 2, 3, 1)

            final_image_bhwc = torch.clamp(final_image_bhwc, 0.0, 1.0)
            final_image_bhwc = self._apply_camera_raw_black_floor_bhwc(final_image_bhwc)

            ALIGNMENT = 32 if enforce_mod32_boundaries else 1
            h_orig, w_orig = final_image_bhwc.shape[1], final_image_bhwc.shape[2]

            if enable_upscale and upscale_factor > 1.0:
                if enforce_mod32_boundaries:
                    target_h, target_w = self._calculate_best_fit_resolution(
                        w_orig, h_orig, upscale_factor, ALIGNMENT
                    )
                else:
                    target_h = int(round(h_orig * upscale_factor))
                    target_w = int(round(w_orig * upscale_factor))

                if upscale_model == "Simple: Bicubic (Standard)":
                    upscaled_temp = F.interpolate(
                        final_image_bhwc.movedim(-1, 1),
                        scale_factor=upscale_factor,
                        mode='bicubic', align_corners=False, antialias=True
                    )
                else:
                    loaded_model = self.load_upscaler_model(upscale_model)
                    if loaded_model:
                        ai_upscaled_image = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']().upscale(
                            upscale_model=loaded_model, image=final_image_bhwc
                        )[0]
                        upscaled_temp = ai_upscaled_image.movedim(-1, 1)
                    else:
                        print(f"Warning: Upscaler model {upscale_model} failed to load. Fallback to bicubic.")
                        upscaled_temp = F.interpolate(
                            final_image_bhwc.movedim(-1, 1),
                            scale_factor=upscale_factor,
                            mode='bicubic', align_corners=False, antialias=True
                        )

                if maintain_aspect_ratio:
                    final_image_bhwc = self._smart_resize_and_crop(
                        upscaled_temp, target_h, target_w
                    ).movedim(1, -1)
                else:
                    final_image_bhwc = F.interpolate(
                        upscaled_temp, size=(target_h, target_w),
                        mode='bicubic', align_corners=False, antialias=True
                    ).movedim(1, -1)

            else:
                if enforce_mod32_boundaries:
                    aligned_h = (h_orig // ALIGNMENT) * ALIGNMENT
                    aligned_w = (w_orig // ALIGNMENT) * ALIGNMENT
                    if aligned_h != h_orig or aligned_w != w_orig:
                        temp = final_image_bhwc.movedim(-1, 1)
                        final_image_bhwc = self._smart_resize_and_crop(
                            temp, aligned_h, aligned_w
                        ).movedim(1, -1)

            final_image_bhwc = torch.clamp(final_image_bhwc, 0.0, 1.0)

            final_latent = None
            if vae is not None:
                if use_tiled_vae:
                    encode_node = nodes.NODE_CLASS_MAPPINGS['VAEEncodeTiled']()
                    encoded = encode_node.encode(
                        vae, final_image_bhwc[:, :, :, :3], tile_size, overlap=64
                    )[0]
                    # VAEEncodeTiled returns {"samples": tensor}, take the tensor
                    final_latent = encoded["samples"] if isinstance(encoded, dict) else encoded
                else:
                    final_latent = vae.encode(final_image_bhwc[:, :, :, :3])
            else:
                final_latent = (
                    latent["samples"] if is_latent_input
                    else torch.zeros((1, 4, 64, 64))
                )

            return ({"samples": final_latent}, final_image_bhwc.cpu())

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"[LatentRefiner] Error: {e}")
            import traceback
            traceback.print_exc()
            dummy_latent = latent if latent is not None else {"samples": torch.zeros((1, 4, 64, 64))}
            dummy_image = image if image is not None else torch.zeros((1, 64, 64, 3))
            return (dummy_latent, dummy_image)
        finally:
            if offload_models_to_cpu:
                # Offload neural corrector via the shared model manager
                try:
                    from .model_manager import ForbiddenVisionModelManager
                    ForbiddenVisionModelManager.get_instance().offload_to_cpu()
                except Exception as e:
                    print(f"[LatentRefiner] Neural corrector offload failed (non-fatal): {e}")

                # Offload depth model if its manager supports it
                try:
                    if hasattr(self.depth_manager, 'offload_to_cpu'):
                        self.depth_manager.offload_to_cpu()
                except Exception as e:
                    print(f"[LatentRefiner] Depth model offload failed (non-fatal): {e}")

    def detect_clipping_issues(self, rgb_bchw):
        luma = 0.2126 * rgb_bchw[:, 0:1] + 0.7152 * rgb_bchw[:, 1:2] + 0.0722 * rgb_bchw[:, 2:3]
        max_rgb = torch.max(rgb_bchw, dim=1, keepdim=True)[0]
        min_rgb = torch.min(rgb_bchw, dim=1, keepdim=True)[0]

        high_luminance = luma > 0.92
        near_clip_max = max_rgb > 0.97
        any_channel_overexposed = torch.any(rgb_bchw > 0.98, dim=1, keepdim=True)
        luminance_clipping = high_luminance & any_channel_overexposed
        
        channel_difference = max_rgb - min_rgb
        has_color = channel_difference > 0.03
        color_only_clipping = near_clip_max & has_color & ~luminance_clipping

        needs_correction = luminance_clipping | color_only_clipping

        return {
            "luminance_clipping": luminance_clipping,
            "color_only_clipping": color_only_clipping,
            "needs_any_correction": needs_correction,
            "luma": luma,
            "max_rgb": max_rgb
        }

    def apply_camera_raw_style_tone_mapping(self, rgb_bchw, analysis):
        try:
            check_for_interruption()
            lum_mask = analysis["luminance_clipping"].float()

            if not torch.any(lum_mask > 0):
                return rgb_bchw

            smooth_mask = kornia.filters.gaussian_blur2d(lum_mask, (11, 11), (2.0, 2.0))
            smooth_mask = torch.clamp(smooth_mask * 1.2, 0, 1)

            luma = analysis["luma"]
            max_rgb = analysis["max_rgb"]

            excess_factor = torch.clamp((max_rgb - 0.95) / 0.05, 0, 1)
            protection = 1.0 - (excess_factor * 0.3)
            scaled = rgb_bchw * protection
            result = torch.where(smooth_mask > 0.01, scaled, rgb_bchw)

            return torch.clamp(result, 0.0, 1.0)
        except model_management.InterruptProcessingException:
            raise
        except Exception:
            return rgb_bchw



    def _apply_camera_raw_black_floor_bhwc(self, img_bhwc, black_floor=1.0/255.0, beta=80.0):
        """
        Ensures no channel hits true 0.0 (which triggers black clipping warnings in ACR/LR),
        using a *soft* floor so it doesn't wash out shadows or create a hard step.

        - black_floor: target minimum (default = 1/255 ≈ 0.00392)
        - beta: softness (higher = closer to hard clamp, but still smooth)
        """
        try:
            x = img_bhwc.float()

            x = black_floor + F.softplus((x - black_floor) * beta) / beta

            return x.clamp(0.0, 1.0).to(img_bhwc.dtype)
        except Exception:
            return torch.clamp(img_bhwc, black_floor, 1.0)

    def apply_final_clipping_protection(self, rgb_bchw):

        try:
            check_for_interruption()
            x = rgb_bchw.float()
            
            threshold = 0.85
            over = x > threshold
            if torch.any(over):
                x_over = x[over]
                x[over] = threshold + (x_over - threshold) / (1.0 + (x_over - threshold))
            
            luma = 0.2126 * x[:,0:1] + 0.7152 * x[:,1:2] + 0.0722 * x[:,2:3]
            max_val = x.max(dim=1, keepdim=True)[0]
            mask = max_val > 1.0
            if torch.any(mask):
                scale = (1.0 - luma) / (max_val - luma + 1e-6)
                x = torch.where(mask, luma + (x - luma) * scale, x)

            return x.clamp(0.0, 1.0).to(rgb_bchw.dtype)
        except Exception:
            return torch.clamp(rgb_bchw, 0.0, 1.0)


    def _apply_neural_correction(self, image_tensor_bhwc, tone_str, color_str):
        """Apply neural color/tone correction (single pass) with integrated vibrance boost."""
        try:
            from .model_manager import ForbiddenVisionModelManager
            manager = ForbiddenVisionModelManager.get_instance()

            input_bchw = image_tensor_bhwc.permute(0, 3, 1, 2).contiguous()

            out_bchw, _ = manager.run_neural_corrector(
                input_bchw,
                tone_strength=tone_str,
                color_strength=color_str,
            )

            output_bhwc = out_bchw.float().permute(0, 2, 3, 1).contiguous()

            print(f"ForbiddenVision: Neural Corrector complete (tone={tone_str:.2f}, color={color_str:.2f})")

            return output_bhwc.to(image_tensor_bhwc.device)

        except Exception as e:
            print(f"[LatentRefiner] Neural corrector failed: {e}")
            import traceback
            traceback.print_exc()
            return image_tensor_bhwc

    def _apply_relight_bchw(self, image_bchw, strength):
        try:
            if self.cached_depth_map is None: return image_bchw
            
            depth = self.cached_depth_map
            if depth.shape[-2:] != image_bchw.shape[-2:]:
                depth = F.interpolate(depth, size=image_bchw.shape[-2:], mode='bilinear', align_corners=False)
                
            mask = depth ** 1.5
            
            lab = kornia.color.rgb_to_lab(image_bchw)
            l, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
            l_norm = l / 100.0
            
            boost = (torch.sin(l_norm * 3.14159) * 0.2) * strength * mask
            l_new = torch.clamp(l_norm + boost, 0, 1) * 100.0
            
            return kornia.color.lab_to_rgb(torch.cat([l_new, a, b], dim=1))
        except:
            return image_bchw

            
    def _apply_dof_depth_only(self, image_bchw, depth_map, dof_strength, depth_dof_focus):
        try:
            check_for_interruption()
            if depth_map is None: return image_bchw

            h, w = image_bchw.shape[-2:]
            if depth_map.shape[-2:] != (h, w):
                depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)

            calibrated_strength = dof_strength ** 1.5
            
            foreground_mask = depth_map > torch.quantile(depth_map, 0.90)
            foreground_depths = depth_map[foreground_mask]
            
            if foreground_depths.numel() > 0:
                focus_plane = torch.mean(foreground_depths)
            else:
                focus_plane = torch.quantile(depth_map, 0.96)
            
            dist = torch.abs(depth_map - focus_plane)
            tolerance = 0.05
            blur_map = torch.clamp((dist - tolerance) / (1.0 - tolerance), 0.0, 1.0)
            blur_map = blur_map * calibrated_strength
            
            max_kernel = 21
            blurred = kornia.filters.gaussian_blur2d(image_bchw, (max_kernel, max_kernel), (10.0, 10.0))
            
            return torch.lerp(image_bchw, blurred, blur_map)

        except Exception as e:
            print(f"DoF error: {e}")
            return image_bchw
        