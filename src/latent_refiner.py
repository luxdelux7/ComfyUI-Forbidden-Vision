import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import nodes
import kornia
from .utils import check_for_interruption, get_ordered_upscaler_model_list

class LatentRefiner:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()

        return {
            "required": {
                "enable_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "upscale_model": (upscaler_models, ),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.05}),
                
                "enable_dynamic_contrast": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "contrast_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 5.0, "step": 0.1}),
                
                "enable_vibrance": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "vibrance_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                "enable_clarity": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "clarity_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.5, "step": 0.05}),
                
                "enable_smart_sharpen": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "sharpening_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                
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

    def load_upscaler_model(self, model_name):
        if self.upscaler_model is not None and self.upscaler_model_name == model_name:
            return self.upscaler_model
        try:
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            self.upscaler_model = upscaler_loader.load_model(model_name)[0]
            self.upscaler_model_name = model_name
            return self.upscaler_model
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error loading upscaler model {model_name}: {e}")
            self.upscaler_model = None; self.upscaler_model_name = None
            return None

    def apply_dynamic_contrast_gpu(self, image_tensor, strength):

        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_dynamic_contrast_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            l_channel = image_lab[:, 0:1, :, :]
            l_normalized = l_channel / 100.0
            
            enhanced_l_normalized = self.apply_simple_s_curve(l_normalized, strength)
            
            enhanced_l = enhanced_l_normalized * 100.0
            
            enhanced_lab = torch.cat([enhanced_l, image_lab[:, 1:3, :, :]], dim=1)
            
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during dynamic contrast enhancement: {e}. Returning original tensor.")
            return image_tensor

    def apply_simple_s_curve(self, luminance, strength):
 
        if strength == 0.0:
            return luminance
        
        image_stats = self.analyze_image_dynamics(luminance)
        
        shadow_adjustment = self.calculate_shadow_adjustment(luminance, image_stats, strength)
        highlight_adjustment = self.calculate_highlight_adjustment(luminance, image_stats, strength)
        midtone_adjustment = self.calculate_midtone_adjustment(luminance, image_stats, strength)
        
        enhanced = luminance + shadow_adjustment + highlight_adjustment + midtone_adjustment
        
        return torch.clamp(enhanced, 0.0, 1.0)
    def analyze_image_dynamics(self, luminance):
  
        mean_luminance = torch.mean(luminance)
        std_luminance = torch.std(luminance)
        
        shadow_amount = torch.mean((luminance < 0.25).float())
        highlight_amount = torch.mean((luminance > 0.75).float())
        midtone_amount = torch.mean(torch.logical_and(luminance >= 0.25, luminance <= 0.75).float())
        
        contrast_level = std_luminance / (mean_luminance + 0.001)
        
        is_low_contrast = contrast_level < 0.3
        
        is_high_contrast = contrast_level > 0.6
        
        is_underexposed = mean_luminance < 0.4
        is_overexposed = mean_luminance > 0.7
        
        return {
            'mean_luminance': mean_luminance,
            'contrast_level': contrast_level,
            'shadow_amount': shadow_amount,
            'highlight_amount': highlight_amount,
            'midtone_amount': midtone_amount,
            'is_low_contrast': is_low_contrast,
            'is_high_contrast': is_high_contrast,
            'is_underexposed': is_underexposed,
            'is_overexposed': is_overexposed
        }
    def calculate_shadow_adjustment(self, luminance, stats, strength):
 
        base_shadow_strength = strength * 0.18
        
        if stats['is_underexposed']:
            shadow_strength = base_shadow_strength * 0.5
        elif stats['shadow_amount'] < 0.1:
            shadow_strength = base_shadow_strength * 1.5
        elif stats['is_low_contrast']:
            shadow_strength = base_shadow_strength * 1.3
        else:
            shadow_strength = base_shadow_strength
        
        shadows_mask = torch.exp(-((luminance - 0.0) / 0.25) ** 2)
        
        shadow_adjustment = -shadow_strength * shadows_mask * (0.25 - luminance).clamp(0, 1)
        
        return shadow_adjustment
    def calculate_highlight_adjustment(self, luminance, stats, strength):
      
        base_highlight_strength = strength * 0.12
        
        if stats['is_overexposed']:
            highlight_strength = base_highlight_strength * 2.0
        elif stats['highlight_amount'] > 0.2:
            highlight_strength = base_highlight_strength * 1.4
        elif stats['highlight_amount'] < 0.05:
            highlight_strength = base_highlight_strength * 0.6
        else:
            highlight_strength = base_highlight_strength
        
        highlights_mask = torch.exp(-((luminance - 1.0) / 0.25) ** 2)
        
        highlight_adjustment = -highlight_strength * highlights_mask * (luminance - 0.75).clamp(0, 1)
        
        return highlight_adjustment
    def calculate_midtone_adjustment(self, luminance, stats, strength):
    
        base_midtone_strength = strength * 0.22
        
        if stats['is_low_contrast']:
            midtone_strength = base_midtone_strength * 1.5
        elif stats['is_high_contrast']:
            midtone_strength = base_midtone_strength * 0.7
        else:
            midtone_strength = base_midtone_strength
        
        midtones_mask = torch.exp(-((luminance - 0.5) / 0.35) ** 2)
        
        midtone_contrast_curve = torch.tanh((luminance - 0.5) * 3.0) * 0.5 + 0.5
        midtone_adjustment = midtone_strength * midtones_mask * (midtone_contrast_curve - luminance)
        
        return midtone_adjustment
    def extract_luminance_channel(self, image_bchw):

        r, g, b = image_bchw[:, 0:1, :, :], image_bchw[:, 1:2, :, :], image_bchw[:, 2:3, :, :]
        
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        return luminance

    def apply_professional_s_curve(self, luminance, strength):

        contrast_factor = (strength - 1.0) * 0.8
        
        shadows_point = 0.25
        highlights_point = 0.75
        midpoint = 0.5
        
        shadows_strength = contrast_factor * 0.7
        highlights_strength = contrast_factor * 0.7
        midtone_strength = contrast_factor * 1.2
        
        shadows_mask = torch.exp(-((luminance - 0.0) / shadows_point) ** 2)
        highlights_mask = torch.exp(-((luminance - 1.0) / (1.0 - highlights_point)) ** 2)
        midtone_mask = torch.exp(-((luminance - midpoint) / 0.3) ** 2)
        
        shadows_adjustment = shadows_strength * shadows_mask * (luminance - shadows_point)
        highlights_adjustment = highlights_strength * highlights_mask * (luminance - highlights_point)
        midtone_adjustment = midtone_strength * midtone_mask * (luminance - midpoint)
        
        enhanced_luminance = luminance + shadows_adjustment + highlights_adjustment + midtone_adjustment
        
        enhanced_luminance = self.apply_smooth_tone_mapping(enhanced_luminance)
        
        return torch.clamp(enhanced_luminance, 0.0, 1.0)

    def apply_smooth_tone_mapping(self, luminance):
 
        over_exposed = torch.clamp(luminance - 1.0, min=0)
        under_exposed = torch.clamp(-luminance, min=0)
        
        smooth_highlights = 1.0 - torch.exp(-over_exposed * 3.0) * 0.15
        smooth_shadows = torch.exp(-under_exposed * 3.0) * 0.15
        
        tone_mapped = luminance - over_exposed + smooth_highlights + under_exposed + smooth_shadows
        
        return tone_mapped

    def preserve_color_saturation(self, original_rgb, enhanced_rgb, strength):
 
        original_luminance = self.extract_luminance_channel(original_rgb)
        enhanced_luminance = self.extract_luminance_channel(enhanced_rgb)
        
        original_max = torch.max(original_rgb, dim=1, keepdim=True)[0]
        original_min = torch.min(original_rgb, dim=1, keepdim=True)[0]
        original_saturation = (original_max - original_min) / (original_max + 1e-8)
        
        enhanced_max = torch.max(enhanced_rgb, dim=1, keepdim=True)[0]
        enhanced_min = torch.min(enhanced_rgb, dim=1, keepdim=True)[0]
        enhanced_saturation = (enhanced_max - enhanced_min) / (enhanced_max + 1e-8)
        
        saturation_preservation = 0.3 * (strength - 1.0)
        target_saturation = original_saturation * (1.0 + saturation_preservation)
        
        saturation_ratio = target_saturation / (enhanced_saturation + 1e-8)
        saturation_ratio = torch.clamp(saturation_ratio, 0.8, 1.3)
        
        luminance_ratio = enhanced_luminance / (original_luminance + 1e-8)
        luminance_ratio = torch.clamp(luminance_ratio, 0.5, 2.0)
        
        preserved_rgb = original_rgb * luminance_ratio
        
        enhanced_center = (enhanced_max + enhanced_min) / 2.0
        preserved_center = (torch.max(preserved_rgb, dim=1, keepdim=True)[0] + 
                        torch.min(preserved_rgb, dim=1, keepdim=True)[0]) / 2.0
        
        final_rgb = preserved_rgb + (enhanced_center - preserved_center) * saturation_ratio
        
        return final_rgb
    
    def apply_vibrance_gpu(self, image_tensor, strength):

        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_vibrance_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            a_channel = image_lab[:, 1:2, :, :]
            b_channel = image_lab[:, 2:3, :, :]
            
            current_saturation = torch.sqrt(a_channel**2 + b_channel**2)
            max_saturation = current_saturation.max()
            
            if max_saturation > 0:
                normalized_saturation = current_saturation / max_saturation
            else:
                normalized_saturation = current_saturation
            
            vibrance_mask = 1.0 - normalized_saturation
            vibrance_mask = torch.pow(vibrance_mask, 0.5)
            
            skin_protection = self.create_skin_tone_mask(a_channel, b_channel)
            final_mask = vibrance_mask * (1.0 - skin_protection * 0.7)
            
            vibrance_factor = 1.0 + (strength * 0.6) * final_mask
            
            enhanced_a = a_channel * vibrance_factor
            enhanced_b = b_channel * vibrance_factor
            
            enhanced_lab = torch.cat([image_lab[:, 0:1, :, :], enhanced_a, enhanced_b], dim=1)
            
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during vibrance enhancement: {e}. Returning original tensor.")
            return image_tensor

    def create_skin_tone_mask(self, a_channel, b_channel):

        skin_a_min, skin_a_max = 0.0, 25.0
        skin_b_min, skin_b_max = 5.0, 35.0
        
        a_in_range = torch.logical_and(a_channel >= skin_a_min, a_channel <= skin_a_max)
        b_in_range = torch.logical_and(b_channel >= skin_b_min, b_channel <= skin_b_max)
        
        skin_mask = torch.logical_and(a_in_range, b_in_range).float()
        
        kernel_size = 5
        skin_mask_blurred = F.avg_pool2d(skin_mask, kernel_size, stride=1, padding=kernel_size//2)
        
        return skin_mask_blurred
    
    def apply_clarity_gpu(self, image_tensor, strength):

        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_clarity_gpu expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            image_bchw = image_tensor.permute(0, 3, 1, 2)
            image_lab = kornia.color.rgb_to_lab(image_bchw)
            
            l_channel = image_lab[:, 0:1, :, :]
            l_normalized = l_channel / 100.0
            
            clarity_detail = self.extract_advanced_clarity_detail(l_normalized, strength)
            
            midtone_mask = self.create_midtone_mask(l_normalized)
            
            adaptive_strength = strength * midtone_mask
            
            enhanced_l_normalized = l_normalized + clarity_detail * adaptive_strength
            enhanced_l_normalized = torch.clamp(enhanced_l_normalized, 0.0, 1.0)
            
            enhanced_l = enhanced_l_normalized * 100.0
            
            enhanced_lab = torch.cat([enhanced_l, image_lab[:, 1:3, :, :]], dim=1)
            
            final_rgb_bchw = kornia.color.lab_to_rgb(enhanced_lab)
            
            return torch.clamp(final_rgb_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error during clarity enhancement: {e}. Returning original tensor.")
            return image_tensor

    def extract_advanced_clarity_detail(self, luminance, strength):
        try:
            check_for_interruption()
            
            device = luminance.device
            dtype = luminance.dtype
            
            base_radius = max(8.0, min(luminance.shape[-1], luminance.shape[-2]) * 0.015)
            base_radius = min(base_radius, 20.0)
            
            if strength > 0.7:
                primary_radius = base_radius * 1.0
                secondary_radius = base_radius * 1.8
                primary_weight = 0.65
                secondary_weight = 0.35
            else:
                primary_radius = base_radius * 1.2
                secondary_radius = None
                primary_weight = 1.0
                secondary_weight = 0.0
            
            primary_blurred = self.advanced_gaussian_blur(luminance, primary_radius, device, dtype)
            primary_detail = (luminance - primary_blurred) * primary_weight
            
            if secondary_radius is not None:
                secondary_blurred = self.advanced_gaussian_blur(luminance, secondary_radius, device, dtype)
                secondary_detail = (luminance - secondary_blurred) * secondary_weight
                combined_detail = primary_detail + secondary_detail
            else:
                combined_detail = primary_detail
            
            edge_mask = self.create_optimized_edge_mask(luminance, device, dtype)
            combined_detail = combined_detail * edge_mask
            
            return combined_detail
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in extract_advanced_clarity_detail: {e}")
            return torch.zeros_like(luminance)

    def create_midtone_mask(self, luminance):
        try:
            check_for_interruption()
            
            distance_from_midtone = torch.abs(luminance - 0.5)
            
            midtone_mask = torch.exp(-distance_from_midtone.pow(2) * 11.111)
            
            midtone_mask = torch.clamp(midtone_mask, 0.2, 1.0)
            
            return midtone_mask
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_midtone_mask: {e}")
            return torch.ones_like(luminance)
    def create_optimized_edge_mask(self, luminance, device, dtype):
        try:
            check_for_interruption()
            
            edge_magnitude = kornia.filters.sobel(luminance)
            
            edge_mask = torch.tanh(edge_magnitude * 6.0)
            edge_mask = torch.clamp(edge_mask, 0.25, 1.0)
            
            return edge_mask
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_optimized_edge_mask: {e}")
            return torch.ones_like(luminance)

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
            print(f"Error in advanced_gaussian_blur: {e}")
            return tensor
    
    def refine_and_process(self, 
                            enable_upscale, upscale_model, upscale_factor,
                            enable_dynamic_contrast, contrast_strength,
                            enable_vibrance, vibrance_strength,
                            enable_clarity, clarity_strength,
                            enable_smart_sharpen, sharpening_strength,
                            use_tiled_vae, tile_size,
                            latent=None, vae=None, image=None, **kwargs):
        try:
            check_for_interruption()
            
            device = model_management.get_torch_device()
            
            if latent is not None and vae is not None:
                decoded_image = vae.decode(latent["samples"].to(device))
                process_latent = True
            elif image is not None:
                decoded_image = image.to(device)
                process_latent = False
            else:
                print("Warning: No valid inputs provided. Please connect either (latent + vae) or image.")
                dummy_latent = {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)}
                dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (dummy_latent, dummy_image)

            image_to_process = decoded_image

            if enable_dynamic_contrast:
                image_to_process = self.apply_dynamic_contrast_gpu(image_to_process, contrast_strength)
                check_for_interruption()

            if enable_vibrance:
                image_to_process = self.apply_vibrance_gpu(image_to_process, vibrance_strength)
                check_for_interruption()

            if enable_clarity:
                image_to_process = self.apply_clarity_gpu(image_to_process, clarity_strength)
                check_for_interruption()

            final_image = image_to_process

            if enable_upscale:
                loaded_model = self.load_upscaler_model(upscale_model)
                if loaded_model is None:
                    print(f"Warning: Upscaler model {upscale_model} failed to load. Skipping upscale.")
                else:
                    ImageUpscalerClass = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']
                    upscaler_node = ImageUpscalerClass()
                    ai_upscaled_image = upscaler_node.upscale(upscale_model=loaded_model, image=image_to_process)[0]
                    check_for_interruption()
                    
                    h, w = image_to_process.shape[1], image_to_process.shape[2]
                    target_h, target_w = int(h * upscale_factor), int(w * upscale_factor)
                    
                    if ai_upscaled_image.shape[1] != target_h or ai_upscaled_image.shape[2] != target_w:
                        final_image = F.interpolate(ai_upscaled_image.movedim(-1, 1), 
                                                    size=(target_h, target_w), 
                                                    mode='bicubic', align_corners=False, antialias=True)
                        final_image = final_image.movedim(1, -1)
                    else:
                        final_image = ai_upscaled_image
            
            check_for_interruption()

            if enable_smart_sharpen and sharpening_strength > 0:
                final_image = self.apply_smart_sharpen(final_image, sharpening_strength)

            check_for_interruption()

            final_image = final_image.to(device)

            if vae is not None:
                if use_tiled_vae:
                    encoder = nodes.NODE_CLASS_MAPPINGS['VAEEncodeTiled']()
                    refined_latent = encoder.encode(vae, final_image, tile_size, 64)[0]
                else:
                    encoder = nodes.NODE_CLASS_MAPPINGS['VAEEncode']()
                    refined_latent = encoder.encode(vae, final_image)[0]

                if not isinstance(refined_latent, dict):
                    refined_latent = {"samples": refined_latent}
            else:
                print("Info: No VAE provided. No latent output available.")
                refined_latent = {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)}

            return (refined_latent, final_image.cpu())

        except model_management.InterruptProcessingException:
            print("Latent Refiner cancelled by user")
            raise
        except Exception as e:
            print(f"Error in Latent Refiner: {e}")
            dummy_latent = {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)}
            dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (dummy_latent, dummy_image)
    
    def apply_smart_sharpen(self, image_tensor, strength):
 
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return image_tensor

            if image_tensor.dim() != 4 or image_tensor.shape[3] != 3:
                print("Warning: apply_smart_sharpen expects a batch of RGB images (BHWC). Skipping.")
                return image_tensor

            device = image_tensor.device
            dtype = image_tensor.dtype
            image_bchw = image_tensor.permute(0, 3, 1, 2)
            
            batch_size, channels, height, width = image_bchw.shape
            edge_fade = self.create_edge_fade_mask(height, width, device, dtype)
            
            gray = 0.299 * image_bchw[:, 0:1] + 0.587 * image_bchw[:, 1:2] + 0.114 * image_bchw[:, 2:3]
            
            blur_small = self.advanced_gaussian_blur(gray, 0.5, device, dtype)
            blur_medium = self.advanced_gaussian_blur(gray, 1.5, device, dtype)
            blur_large = self.advanced_gaussian_blur(gray, 3.0, device, dtype)
            
            detail_fine = gray - blur_small
            detail_medium = gray - blur_medium  
            detail_coarse = gray - blur_large
            
            detail_magnitude = torch.abs(detail_medium)
            edge_mask = torch.tanh(detail_magnitude * 20.0)
            
            edge_mask = edge_mask * edge_fade
            
            smooth_mask = torch.tanh(detail_magnitude * 5.0)
            edge_mask = edge_mask * (0.3 + 0.7 * smooth_mask)
            
            if strength <= 0.5:
                combined_detail = detail_fine * 0.7 + detail_medium * 0.3
                multiplier = strength * 4.0
            elif strength <= 1.0:
                combined_detail = detail_fine * 0.4 + detail_medium * 0.4 + detail_coarse * 0.2
                multiplier = 2.0 + (strength - 0.5) * 6.0
            else:
                combined_detail = detail_fine * 0.2 + detail_medium * 0.5 + detail_coarse * 0.3
                multiplier = 5.0 + (strength - 1.0) * 10.0
            
            for c in range(3):
                channel = image_bchw[:, c:c+1]
                channel_blur = self.advanced_gaussian_blur(channel, 1.0, device, dtype)
                channel_detail = channel - channel_blur
                
                sharpening = channel_detail * edge_mask * multiplier
                image_bchw[:, c:c+1] = channel + sharpening
            
            return torch.clamp(image_bchw, 0.0, 1.0).permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smart sharpen: {e}")
            return image_tensor
  
  
    def create_edge_fade_mask(self, height, width, device, dtype):
   
        size_factor = min(height, width) / 512.0
        fade_percentage = 0.02 + (0.01 * min(1.0, size_factor / 4.0))
        fade_distance = min(height, width) * fade_percentage
        
        y_profile = torch.ones(height, device=device, dtype=dtype)
        x_profile = torch.ones(width, device=device, dtype=dtype)
        
        for i in range(int(fade_distance)):
            fade_value = i / fade_distance
            y_profile[i] = fade_value
            y_profile[-(i+1)] = fade_value
            x_profile[i] = fade_value
            x_profile[-(i+1)] = fade_value
        
        mask_2d = y_profile.unsqueeze(1) * x_profile.unsqueeze(0)
        
        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
        
        return mask_4d
   
   
    def smooth_mask(self, mask, device):

        kernel_size = 3
        smooth_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=device) / (kernel_size * kernel_size)
        
        smoothed = F.conv2d(mask, smooth_kernel, padding=kernel_size//2)
        
        return smoothed


    def get_color_matrices(self, device, dtype):
        if not hasattr(self, '_color_cache'): self._color_cache = {}
        cache_key = (device, dtype)
        if cache_key in self._color_cache: return self._color_cache[cache_key]
        rgb_to_ycbcr_matrix = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=dtype, device=device).t()
        ycbcr_to_rgb_matrix = torch.tensor([[1., 0., 1.402], [1., -0.344136, -0.714136], [1., 1.772, 0.]], dtype=dtype, device=device).t()
        bias = torch.tensor([0., 0.5, 0.5], dtype=dtype, device=device)
        self._color_cache[cache_key] = (rgb_to_ycbcr_matrix, ycbcr_to_rgb_matrix, bias)
        return rgb_to_ycbcr_matrix, ycbcr_to_rgb_matrix, bias

    def create_gaussian_kernel(self, kernel_size, sigma, device, dtype):
        if not hasattr(self, '_kernel_cache'): 
            self._kernel_cache = {}
        
        cache_key = (kernel_size, sigma, str(device), str(dtype))
        
        if cache_key in self._kernel_cache: 
            return self._kernel_cache[cache_key]
        
        try:
            check_for_interruption()
            
            x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
            gauss = torch.exp(-0.5 * (x / sigma).pow(2))
            kernel_1d = gauss / gauss.sum()
            kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None, :, :]
            
            self._kernel_cache[cache_key] = kernel
            
            if len(self._kernel_cache) > 15: 
                oldest_key = next(iter(self._kernel_cache))
                del self._kernel_cache[oldest_key]
            
            return kernel
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_gaussian_kernel: {e}")
            return torch.ones((1, 1, 1, 1), dtype=dtype, device=device)

    def rgb_to_ycbcr(self, image_rgb):
        rgb_to_ycbcr_matrix, _, bias = self.get_color_matrices(image_rgb.device, image_rgb.dtype)
        return torch.matmul(image_rgb, rgb_to_ycbcr_matrix) + bias

    def ycbcr_to_rgb(self, image_ycbcr):
        _, ycbcr_to_rgb_matrix, bias = self.get_color_matrices(image_ycbcr.device, image_ycbcr.dtype)
        return torch.matmul(image_ycbcr - bias, ycbcr_to_rgb_matrix)