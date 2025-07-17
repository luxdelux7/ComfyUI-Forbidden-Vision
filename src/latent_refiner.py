import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import nodes
import kornia
from PIL import Image
from .utils import check_for_interruption, get_ordered_upscaler_model_list
try:
    from transparent_background import Remover
except ImportError:
    print("-------------------------------------------------------------------------------------------------")
    print("WARNING: transparent-background library not found.")
    print("Please install it by adding 'transparent-background' to your requirements.txt and running the update script.")
    print("The Depth of Field (DOF) effect will not be available until this is installed.")
    print("-------------------------------------------------------------------------------------------------")
class LatentRefiner:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()
        
        simple_method = "Simple: Bicubic (Standard)"
        upscaler_models.insert(0, simple_method)

        return {
            "required": {
                "operating_mode": (["General", "Anime"],),

                "enable_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "upscale_model": (upscaler_models, ),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.05}),
                
                "enable_dof_effect": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "dof_blur_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                
                "relighting_mode": (["Disabled", "Additive (Simple)", "Corrective"],),
                "relight_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mood_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mood_background_replace": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),

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
                "enable_latent_refinement": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "latent_refinement_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mood_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "refined_image_preview")
    FUNCTION = "refine_and_process"
    CATEGORY = "Forbidden Vision"

    def __init__(self):
        self.upscaler_model = None
        self.upscaler_model_name = None
        self.remover = None

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
            
    def refine_and_process(self,
                        operating_mode,
                        enable_upscale, upscale_model, upscale_factor,
                        enable_dof_effect, dof_blur_strength,
                        relighting_mode, relight_strength,
                        mood_strength, mood_background_replace,
                        enable_dynamic_contrast, contrast_strength,
                        enable_vibrance, vibrance_strength,
                        enable_clarity, clarity_strength,
                        enable_smart_sharpen, sharpening_strength,
                        use_tiled_vae, tile_size,
                        enable_latent_refinement, latent_refinement_strength,
                        latent=None, vae=None, image=None, mood_image=None, **kwargs):
        try:
            refined_latent = latent
            check_for_interruption()

            device = model_management.get_torch_device()

            if latent is not None and vae is not None:
                decoded_image = vae.decode(latent["samples"].to(device))
            elif image is not None:
                decoded_image = image.to(device)
            else:
                print("Warning: No valid inputs provided. Please connect either (latent + vae) or image.")
                dummy_latent = {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)}
                dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (dummy_latent, dummy_image)

            image_to_process = decoded_image

            if enable_dof_effect or relighting_mode != "Disabled" or (mood_image is not None and mood_strength > 0):
                image_to_process = self.apply_segmentation_effects(
                    image_to_process,
                    operating_mode,
                    enable_dof_effect, dof_blur_strength,
                    relighting_mode, relight_strength,
                    mood_image, mood_strength, mood_background_replace
                )
                check_for_interruption()

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

            if enable_upscale and upscale_factor > 1.0:
                h, w = image_to_process.shape[1], image_to_process.shape[2]
                target_h, target_w = int(h * upscale_factor), int(w * upscale_factor)

                if upscale_model == "Simple: Bicubic (Standard)":
                    print(f"Refiner: Using Simple Upscale (Bicubic) from {w}x{h} to {target_w}x{target_h}")
                    final_image = F.interpolate(image_to_process.movedim(-1, 1),
                                                size=(target_h, target_w),
                                                mode='bicubic',
                                                align_corners=False,
                                                antialias=True)
                    final_image = final_image.movedim(1, -1)
                else:
                    print(f"Refiner: Using AI Model '{upscale_model}' to upscale image.")
                    loaded_model = self.load_upscaler_model(upscale_model)
                    if loaded_model is None:
                        print(f"Warning: Upscaler model {upscale_model} failed to load. Skipping upscale.")
                        final_image = image_to_process
                    else:
                        ImageUpscalerClass = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']
                        upscaler_node = ImageUpscalerClass()
                        ai_upscaled_image = upscaler_node.upscale(upscale_model=loaded_model, image=image_to_process)[0]
                        check_for_interruption()

                        if ai_upscaled_image.shape[1] != target_h or ai_upscaled_image.shape[2] != target_w:
                            final_image = F.interpolate(ai_upscaled_image.movedim(-1, 1),
                                                        size=(target_h, target_w),
                                                        mode='bicubic', align_corners=False, antialias=True)
                            final_image = final_image.movedim(1, -1)
                        else:
                            final_image = ai_upscaled_image

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

                if enable_latent_refinement:
                    enhanced_samples = self.apply_latent_refinement(
                        refined_latent["samples"], latent_refinement_strength
                    )
                    refined_latent["samples"] = enhanced_samples
                    check_for_interruption()

            return (refined_latent, final_image.cpu())

        except model_management.InterruptProcessingException:
            print("Refiner cancelled by user")
            raise
        except Exception as e:
            print(f"Error in Refiner: {e}")
            import traceback
            traceback.print_exc()
            dummy_latent = {"samples": torch.zeros((1, 4, 64, 64), dtype=torch.float32)} if latent is None else latent
            dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32) if image is None else image
            return (dummy_latent, dummy_image)

    def _analyze_mood_image(self, mood_image_bchw, device):
        """
        Performs a definitive, multi-tone color analysis using a k-means approximation
        to find the most dominant color in each tonal range for maximum accuracy.
        """
        try:
            with torch.no_grad():
                mood_gray = kornia.color.rgb_to_grayscale(mood_image_bchw)
                
                def get_dominant_color(image_bchw, mask):
                    # Check if there are enough pixels to analyze
                    if mask.sum() < 100:
                        return None
                    
                    # Apply mask and prepare pixels for analysis
                    pixels = image_bchw * mask
                    pixels = pixels.permute(0, 2, 3, 1).reshape(-1, 3)
                    valid_pixels = pixels[mask.reshape(-1), :]

                    if valid_pixels.shape[0] < 3: # Not enough pixels for k-means
                        return torch.mean(valid_pixels, dim=0)

                    # Simple k-means approximation to find the dominant color cluster
                    # We use a small number of iterations for speed
                    centroids = valid_pixels[torch.randperm(valid_pixels.shape[0])[:1]]
                    for _ in range(5):
                        distances = torch.cdist(valid_pixels, centroids)
                        labels = torch.argmin(distances, dim=1)
                        # Check for empty clusters before calculating the mean
                        if torch.bincount(labels).shape[0] > 0:
                            new_centroids = torch.stack([valid_pixels[labels == i].mean(dim=0) for i in range(1) if (labels == i).any()])
                            if new_centroids.shape[0] == 1:
                                centroids = new_centroids
                    return centroids[0]

                # 1. Highlight Color Analysis
                highlight_mask = mood_gray >= 0.7
                accent_color = get_dominant_color(mood_image_bchw, highlight_mask)
                
                # 2. Mid-tone Color Analysis
                mid_tone_mask = (mood_gray >= 0.3) & (mood_gray < 0.7)
                main_color = get_dominant_color(mood_image_bchw, mid_tone_mask)

                # 3. Shadow Color Analysis
                shadow_mask = mood_gray < 0.3
                shadow_color = get_dominant_color(mood_image_bchw, shadow_mask)

                # Fallbacks if a specific tonal range is missing
                if main_color is None: main_color = torch.mean(mood_image_bchw, dim=(2,3)).squeeze()
                if accent_color is None: accent_color = torch.clamp(main_color * 1.4, 0, 1)
                if shadow_color is None: shadow_color = torch.clamp(main_color * 0.6, 0, 1)

                return {
                    "main_color": main_color,
                    "accent_color": accent_color,
                    "shadow_color": shadow_color,
                }
                
        except Exception as e:
            print(f"Error in dominant color analysis: {e}")
            return {
                "main_color": torch.tensor([0.5, 0.5, 0.5], device=device),
                "accent_color": torch.tensor([0.8, 0.8, 0.8], device=device),
                "shadow_color": torch.tensor([0.2, 0.2, 0.2], device=device)
            }
    def _create_influence_zones(self, subject_mask, h, w, device):
        """
        Creates different influence zones for mood transfer:
        - Core: Minimal influence (preserve character)
        - Edge: Medium influence (blend with environment)  
        - Background: Full influence
        """
        try:
            check_for_interruption()
            
            scale_factor = min(h, w) / 1024.0
            
            # Create a softened mask for general compositing and background calculations
            edge_blur = max(8.0 * scale_factor, 2.0)
            edge_kernel_size = int(edge_blur * 2) | 1
            soft_subject_mask = kornia.filters.gaussian_blur2d(
                subject_mask, 
                (edge_kernel_size, edge_kernel_size), 
                (edge_blur, edge_blur)
            )

            # Create core mask (eroded subject mask) to protect the character's center
            core_erosion = max(int(10 * scale_factor), 2) # Slightly larger erosion
            core_kernel_size = core_erosion * 2 + 1
            core_kernel = torch.ones(core_kernel_size, core_kernel_size, device=device)
            core_mask = kornia.morphology.erosion(soft_subject_mask, core_kernel)
            
            # Edge zone is the transition area between the core and the full background
            edge_mask = torch.clamp(soft_subject_mask - core_mask, 0, 1)
            
            # Background is everything not covered by the soft subject mask
            background_mask = 1.0 - soft_subject_mask
            
            return {
                'core_mask': core_mask,
                'edge_mask': edge_mask, 
                'background_mask': background_mask,
                'soft_subject_mask': soft_subject_mask
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating influence zones: {e}")
            # Fallback to a simple subject/background split with a soft edge
            edge_blur = max(8.0 * min(h, w) / 1024.0, 2.0)
            edge_kernel_size = int(edge_blur * 2) | 1
            soft_mask = kornia.filters.gaussian_blur2d(subject_mask, (edge_kernel_size, edge_kernel_size), (edge_blur, edge_blur))
            
            return {
                'core_mask': subject_mask,
                'edge_mask': torch.zeros_like(subject_mask), # No separate edge in simple fallback
                'background_mask': 1.0 - soft_mask,
                'soft_subject_mask': soft_mask
            }
    
    def _blend_color_lab(self, image_bchw, target_color_rgb, blend_strength):
        """
        Blends a target color onto an image in the LAB color space.
        This preserves the original image's luminosity, preventing it from getting
        washed out or overblown, while effectively transferring the color mood.

        Args:
            image_bchw: The source image tensor (Batch, Channels, Height, Width).
            target_color_rgb: A torch tensor for the target color, e.g., torch.tensor([r, g, b]).
            blend_strength: How strongly to apply the color (0.0 to 1.0).

        Returns:
            The color-blended image tensor in RGB format.
        """
        try:
            # Convert source image to LAB
            source_lab = kornia.color.rgb_to_lab(image_bchw)
            
            # Create a full image of the target color and convert it to LAB
            target_color_image = target_color_rgb.view(1, 3, 1, 1).expand_as(image_bchw)
            target_lab = kornia.color.rgb_to_lab(target_color_image)
            target_a = target_lab[:, 1:2]
            target_b = target_lab[:, 2:3]
            
            # Blend the 'a' and 'b' (color) channels
            # We keep the original 'L' (lightness) channel from the source image
            blended_a = torch.lerp(source_lab[:, 1:2], target_a, blend_strength)
            blended_b = torch.lerp(source_lab[:, 2:3], target_b, blend_strength)
            
            # Reconstruct the LAB image with original lightness and new colors
            final_lab = torch.cat([source_lab[:, 0:1], blended_a, blended_b], dim=1)
            
            # Convert back to RGB and clamp
            return torch.clamp(kornia.color.lab_to_rgb(final_lab), 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _blend_color_lab: {e}")
            return image_bchw
    def _apply_atmospheric_glow_effects(self, image_bchw, mood_analysis, influence_zones, device):
        """
        Applies atmospheric glow effects using mood accent colors.
        This function generates the full-strength (strength=1.0) version of the effect.
        """
        try:
            check_for_interruption()
            
            h, w = image_bchw.shape[-2:]
            scale_factor = min(h, w) / 1024.0
            
            result = image_bchw.clone()
            accent_color = mood_analysis['accent_color'].view(1, 3, 1, 1)
            
            # 1. Ambient Glow (environmental lighting)
            # This should subtly tint the entire scene, including the character's edges.
            ambient_strength = 0.20 # Increased from 0.08
            ambient_radius = max(25.0 * scale_factor, 10.0)
            ambient_kernel_size = int(ambient_radius * 2) | 1
            
            # The glow should emanate from the background and wrap around the subject
            ambient_map = kornia.filters.gaussian_blur2d(
                1.0 - influence_zones['soft_subject_mask'], 
                (ambient_kernel_size, ambient_kernel_size), 
                (ambient_radius, ambient_radius)
            )
            
            # Apply ambient glow with zone influence, affecting the character's edges and background
            ambient_influence_mask = (influence_zones['core_mask'] * 0.2 + # Affects core slightly
                                        influence_zones['edge_mask'] * 0.7 + 
                                        influence_zones['background_mask'] * 1.0)
            
            ambient_glow = ambient_map * accent_color * ambient_strength * ambient_influence_mask
            # Use screen blend for light addition
            result = 1.0 - (1.0 - result) * (1.0 - ambient_glow)
            
            # 2. Highlight Enhancement (bloom/glow from bright spots)
            original_gray = kornia.color.rgb_to_grayscale(image_bchw)
            highlight_threshold = 0.7
            highlight_mask = torch.clamp((original_gray - highlight_threshold) / (1.0 - highlight_threshold), 0, 1)
            
            # Blur highlights for glow effect
            highlight_blur = max(15.0 * scale_factor, 5.0)
            highlight_kernel_size = int(highlight_blur * 2) | 1
            soft_highlight_mask = kornia.filters.gaussian_blur2d(
                highlight_mask, 
                (highlight_kernel_size, highlight_kernel_size), 
                (highlight_blur, highlight_blur)
            )
            
            # Define how strongly the highlight glow affects each zone
            highlight_core_strength = 0.25 # Increased from 0.15
            highlight_edge_strength = 0.40 # Increased from 0.25
            highlight_bg_strength = 0.55 # Increased from 0.35
            
            highlight_influence = (influence_zones['core_mask'] * highlight_core_strength +
                                    influence_zones['edge_mask'] * highlight_edge_strength +
                                    influence_zones['background_mask'] * highlight_bg_strength)
            
            highlight_effect = soft_highlight_mask * accent_color * highlight_influence
            # Use screen blend for light addition
            result = 1.0 - (1.0 - result) * (1.0 - highlight_effect)
            
            # 3. Rim lighting (subtle edge enhancement on the subject)
            edge_kernel_size = max(3, int(3 * scale_factor) | 1)
            # Use a Sobel filter for cleaner edge detection from the soft mask
            edge_mask = kornia.filters.sobel(influence_zones['soft_subject_mask'])

            # Soften rim light
            rim_blur = max(6.0 * scale_factor, 2.0)
            rim_kernel_size = int(rim_blur * 2) | 1
            soft_rim_mask = kornia.filters.gaussian_blur2d(
                edge_mask, 
                (rim_kernel_size, rim_kernel_size), 
                (rim_blur, rim_blur)
            )
            
            # Apply rim lighting
            rim_strength = 0.45 # Increased from 0.2
            rim_effect = soft_rim_mask * accent_color * rim_strength
            result = 1.0 - (1.0 - result) * (1.0 - rim_effect)
            
            return torch.clamp(result, 0.0, 1.0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying atmospheric glow effects: {e}")
            return image_bchw
    def _apply_conservative_color_grading(self, image_bchw, mood_analysis, influence_zones, device):
        """
        Apply mood colors to make the image belong in that mood.
        This function generates the full-strength (strength=1.0) version of the effect.
        """
        try:
            check_for_interruption()
            
            main_color = mood_analysis['main_color']
            temperature = mood_analysis['temperature']
            brightness = mood_analysis['brightness']
            
            result = image_bchw.clone()
            
            core_mask = influence_zones['core_mask']
            edge_mask = influence_zones['edge_mask']
            bg_mask = influence_zones['background_mask']
            
            # Define the maximum strength of the effect for different image zones.
            # This creates the "gentle but visible" effect on the character.
            core_strength = 0.35  # Increased from 0.15
            edge_strength = 0.65  # Increased from 0.4
            bg_strength = 0.95   # Increased from 0.8
            
            # Create a single influence map from the zones and their strengths
            color_influence = (core_mask * core_strength + 
                                edge_mask * edge_strength + 
                                bg_mask * bg_strength)
            
            # Apply mood color tint
            # We use a soft light blend for a more natural color transfer
            mood_color_layer = main_color.view(1, 3, 1, 1).expand_as(result)
            colored_image = self._soft_light_blend(result, mood_color_layer)
            result = torch.lerp(result, colored_image, color_influence * 0.6)

            # Apply temperature shift
            if abs(temperature) > 0.05:
                # Create a warm/cool filter based on temperature
                temp_shift_color = torch.tensor([temperature, 0, -temperature], device=device).view(1, 3, 1, 1) * 0.5
                # Apply with a soft overlay to avoid unnatural color casts
                result = torch.lerp(result, torch.clamp(result + temp_shift_color, 0, 1), color_influence)

            # Brightness matching
            # This part can easily wash out details, so we keep it subtle
            current_brightness = torch.mean(kornia.color.rgb_to_grayscale(result) * influence_zones['soft_subject_mask'])
            if current_brightness > 0:
                brightness_diff = brightness - current_brightness
                # Apply brightness change gently, mostly to background
                brightness_influence = (influence_zones['core_mask'] * 0.1 +
                                        influence_zones['edge_mask'] * 0.2 +
                                        influence_zones['background_mask'] * 0.8)
                result = torch.clamp(result + (brightness_diff * brightness_influence), 0.0, 1.0)
            
            return torch.clamp(result, 0.0, 1.0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in color grading: {e}")
            return image_bchw
    
    def _generate_procedural_light_map(self, background_bchw, device):
        """
        Analyzes the background to find the primary light source's location,
        then generates a clean, procedural gradient map representing that light.
        This avoids projecting the background's texture onto the foreground.
        """
        try:
            h, w = background_bchw.shape[-2:]
            
            # 1. Find the brightest point in the background to use as a reference
            bg_gray = kornia.color.rgb_to_grayscale(background_bchw)
            # Find the index of the max value
            brightest_index = torch.argmax(bg_gray)
            # Convert the flat index to 2D coordinates
            brightest_y = (brightest_index // w) / h
            brightest_x = (brightest_index % w) / w

            # 2. Create a coordinate grid
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(0, 1, h, device=device),
                torch.linspace(0, 1, w, device=device),
                indexing='ij'
            )

            # 3. Calculate distance from every pixel to the brightest point
            distance = torch.sqrt((x_coords - brightest_x)**2 + (y_coords - brightest_y)**2)
            
            # 4. Generate a smooth falloff gradient based on distance
            falloff_radius = 0.75 # Controls the "size" of the light source
            falloff_exponent = 1.5 # Controls the softness of the light's edge
            light_map = 1.0 / (1.0 + (distance / falloff_radius).pow(falloff_exponent))
            
            return light_map.unsqueeze(0).unsqueeze(0)

        except Exception as e:
            print(f"Error generating procedural light map: {e}")
            return torch.zeros((1, 1, h, w), device=device)
    
    def _suppress_spill(self, foreground_bchw, background_bchw, subject_mask, scale_factor):
        """
        Applies a definitive, distance-based spill suppression to create a seamless edge blend.
        It neutralizes the character's edge and tints it with the local background color.
        """
        try:
            # 1. Create a true, inward-fading distance map.
            # We simulate a distance transform by heavily blurring the inside of the mask.
            # The blur amount is dynamic and controls the thickness of the blend zone.
            blend_thickness = max(8.0 * scale_factor, 4.0)
            kernel_size = int(blend_thickness * 2) | 1
            
            # A big box blur is a fast and effective way to create a soft interior gradient
            blurred_interior = kornia.filters.box_blur(subject_mask, (kernel_size, kernel_size))
            
            # The distance map is the difference, revealing a soft gradient from the edge inward.
            distance_map = torch.clamp((subject_mask - blurred_interior) * 1.5, 0, 1)

            if distance_map.max() == 0:
                return foreground_bchw

            # 2. Desaturate the foreground to create a neutral color base
            fg_desaturated = kornia.color.rgb_to_grayscale(foreground_bchw).repeat(1, 3, 1, 1)
            
            # 3. Get the local background color to tint the edge with
            ambient_edge_color = kornia.filters.gaussian_blur2d(background_bchw, (kernel_size, kernel_size), (blend_thickness, blend_thickness))

            # 4. Perform the two-stage edge treatment
            # First, neutralize the original edge color by blending towards the desaturated version
            neutralized_fg = torch.lerp(foreground_bchw, fg_desaturated, distance_map)
            # Second, tint that neutral edge with the new background's ambient color
            tinted_fg = torch.lerp(neutralized_fg, ambient_edge_color, distance_map * 0.75) # *0.75 to keep it subtle
            
            # The final result is the original foreground, with only its edges replaced by the treated version
            return torch.lerp(foreground_bchw, tinted_fg, distance_map)

        except Exception as e:
            print(f"Error in _suppress_spill: {e}")
            return foreground_bchw
    def apply_mood_and_lighting_transfer(self, image_bchw, subject_mask, mood_image, strength, mood_background_replace):
        """
        Applies a definitive, multi-stage mood and lighting transfer that correctly
        grades the background and integrates the foreground for a final, professional result.
        """
        try:
            check_for_interruption()
            device = image_bchw.device
            h, w = image_bchw.shape[-2:]
            scale_factor = min(h, w) / 1024.0
            original_image_bchw = image_bchw.clone()

            if mood_image.dim() == 3: 
                mood_image = mood_image.unsqueeze(0)
            mood_image_bchw_orig = mood_image.permute(0, 3, 1, 2).to(device)
            
            # --- Stage 1: Advanced Scene & Mood Analysis ---
            mood_analysis_image = self._create_mood_background(mood_image_bchw_orig, h, w)
            mood_palette = self._analyze_mood_image(mood_analysis_image, device)
            
            # --- Stage 2: Background Plate Preparation (LOGIC FIX) ---
            # This is the critical change. We now correctly prepare the background plate
            # that will be used in the final composite.
            if mood_background_replace:
                background_plate = mood_analysis_image
            else:
                # When not replacing, the plate IS the color-graded original background.
                # This ensures the background always shifts to match the mood.
                background_plate = self._blend_color_lab(original_image_bchw, mood_palette['main_color'], 0.8)

            # --- Stage 3: Photographic Foreground Relighting ---
            # The foreground is always relit based on the original character image.
            relit_foreground = original_image_bchw.clone()

            fg_luminance = kornia.color.rgb_to_grayscale(relit_foreground)
            highlight_map = fg_luminance.pow(2.5) * subject_mask
            shadow_map = (1.0 - fg_luminance).pow(2.5) * subject_mask
            procedural_light_map = self._generate_procedural_light_map(background_plate, device)

            relit_foreground = self._blend_color_lab(relit_foreground, mood_palette['main_color'], 0.3)
            
            shadow_color_layer = mood_palette['shadow_color'].view(1, 3, 1, 1).expand_as(relit_foreground)
            tinted_shadows = self._soft_light_blend(relit_foreground, shadow_color_layer)
            relit_foreground = torch.lerp(relit_foreground, tinted_shadows, shadow_map * 0.75)

            highlight_color_layer = mood_palette['accent_color'].view(1, 3, 1, 1).expand_as(relit_foreground)
            tinted_highlights = self._soft_light_blend(relit_foreground, highlight_color_layer)
            relit_foreground = torch.lerp(relit_foreground, tinted_highlights, (highlight_map * procedural_light_map) * 0.85)
            
            # --- Stage 4: Spill Suppression & Final Composite (LOGIC FIX) ---
            spill_suppressed_foreground = self._suppress_spill(relit_foreground, background_plate, subject_mask, scale_factor)

            feather_amount = max(2.0 * scale_factor, 1.0)
            kernel_size = int(feather_amount * 2) | 1
            soft_mask = kornia.filters.gaussian_blur2d(subject_mask, (kernel_size, kernel_size), (feather_amount, feather_amount))
            final_compositing_mask = torch.max(subject_mask, soft_mask)

            # The composite is now correctly built using the prepared background_plate.
            final_image_processed = torch.lerp(background_plate, spill_suppressed_foreground, final_compositing_mask)

            # --- Stage 5: Final Blend ---
            # If the background was replaced, we blend from the original image to the new composite.
            # If the background was graded, we also blend from the original to the new composite.
            # This behavior is correct and consistent.
            final_strength = pow(strength, 1.2)
            result = torch.lerp(original_image_bchw, final_image_processed, final_strength)
            
            return torch.clamp(result, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying mood and lighting transfer: {e}")
            import traceback
            traceback.print_exc()
            return image_bchw
    def _construct_lit_foreground(self, original_bchw, subject_mask, ambient_color, highlight_color, lighting_map, scale_factor):
        """
        Applies bounce and rim light using a Screen blend and then correctly transfers
        the resulting hue to the character using the LAB color space.
        """
        device = original_bchw.device
        
        relit_reference = original_bchw.clone()

        bounce_light_effect = torch.lerp(ambient_color, highlight_color, lighting_map) * 0.7
        relit_reference = 1.0 - (1.0 - relit_reference) * (1.0 - bounce_light_effect)

        edge_kernel_size = max(3, int(3 * scale_factor) | 1)
        edge_mask = kornia.morphology.gradient(subject_mask, torch.ones(edge_kernel_size, edge_kernel_size, device=device))
        rim_blur_radius = max(6.0 * scale_factor, 3.0)
        rim_kernel_size = int(rim_blur_radius * 2) | 1
        soft_rim_mask = kornia.filters.gaussian_blur2d(edge_mask, (rim_kernel_size, rim_kernel_size), (rim_blur_radius, rim_blur_radius))
        rim_light_effect = soft_rim_mask * highlight_color * 1.2
        relit_reference = 1.0 - (1.0 - relit_reference) * (1.0 - rim_light_effect)
        
        original_lab = kornia.color.rgb_to_lab(original_bchw)
        relit_lab = kornia.color.rgb_to_lab(relit_reference)
        
        final_lab = torch.cat([original_lab[:, 0:1, :, :], relit_lab[:, 1:3, :, :]], dim=1)
        
        return kornia.color.lab_to_rgb(final_lab)

    def _apply_highlight_glow(self, original_bchw, background, foreground, highlight_color, scale_factor):
        """Finds highlights in the original image and applies a colored glow."""
        original_gray = kornia.color.rgb_to_grayscale(original_bchw)
        highlight_mask = torch.clamp((original_gray - 0.85) / 0.15, 0, 1)
        glow_blur = max(8.0 * scale_factor, 4.0)
        glow_kernel = int(glow_blur * 2) | 1
        soft_highlight_mask = kornia.filters.gaussian_blur2d(highlight_mask, (glow_kernel, glow_kernel), (glow_blur, glow_blur))
        glow_effect = soft_highlight_mask * highlight_color
        
        background = 1.0 - (1.0 - background) * (1.0 - glow_effect * 0.8)
        foreground = 1.0 - (1.0 - foreground) * (1.0 - glow_effect * 0.4)
        return background, foreground
    def _create_mood_background(self, mood_image_bchw, target_h, target_w):
        """
        Performs a high-quality, aspect-ratio-preserving 'cover' resize of the
        mood image to be used as a replacement background.
        """
        mood_h, mood_w = mood_image_bchw.shape[-2:]
        
        mood_aspect = mood_w / mood_h
        target_aspect = target_w / target_h
        
        if mood_aspect > target_aspect:
            new_h = target_h
            new_w = int(target_h * mood_aspect)
        else:
            new_w = target_w
            new_h = int(target_w / mood_aspect)
            
        interp_mode = 'area' if (new_w * new_h) < (mood_w * mood_h) else 'bicubic'
        resized_mood = F.interpolate(mood_image_bchw, size=(new_h, new_w), mode=interp_mode)
        
        y_offset = (new_h - target_h) // 2
        x_offset = (new_w - target_w) // 2
        return resized_mood[:, :, y_offset:y_offset+target_h, x_offset:x_offset+target_w]    
    def get_mask_bounding_box(self, mask):
        """Finds the bounding box of the non-zero regions in a mask tensor."""
        if mask.sum() == 0:
            return 0, 0, mask.shape[3], mask.shape[2]
        
        rows = torch.any(mask, dim=3).squeeze()
        cols = torch.any(mask, dim=2).squeeze()
        
        y_min, y_max = torch.where(rows)[0][[0, -1]]
        x_min, x_max = torch.where(cols)[0][[0, -1]]
        
        return x_min.item(), y_min.item(), x_max.item(), y_max.item()
   
    def _apply_dof_pyramid(self, image_bchw, subject_mask, dof_strength, device):
        """
        Internal helper to apply the high-quality, pyramid-based depth of field effect.
        """
        h, w = image_bchw.shape[-2:]
        scale_factor = min(h, w) / 1024.0
        
        contract_k = torch.ones(max(1, int(3*scale_factor))*2+1, max(1, int(3*scale_factor))*2+1, device=device)
        eroded_mask = kornia.morphology.erosion(subject_mask, contract_k)
        
        map_blur = (dof_strength * 1.5 + 2.0) * scale_factor
        blur_map = torch.pow(1.0 - kornia.filters.gaussian_blur2d(eroded_mask, (int(map_blur*2)|1, int(map_blur*2)|1), (map_blur, map_blur)), 3.0)
        
        num_levels = 4
        pyramid = [image_bchw]
        for i in range(1, num_levels):
            level_strength = (dof_strength * scale_factor) * (i / (num_levels - 1))
            k_size = int(level_strength*2)|1
            if k_size < 3: pyramid.append(image_bchw)
            else: pyramid.append(kornia.filters.gaussian_blur2d(image_bchw, (k_size, k_size), (level_strength, level_strength)))
        
        scaled_map = blur_map * (num_levels - 1)
        background = pyramid[0]
        for i in range(num_levels - 1):
            background = torch.lerp(background, pyramid[i+1], torch.clamp(scaled_map - i, 0.0, 1.0))
            
        return background
    def _soft_light_blend(self, bottom, top):
        """
        Applies a Soft Light blend mode between two tensors.
        The tensors are expected to be in the range [0, 1].
        """
        return torch.where(top > 0.5,
                            1.0 - (1.0 - 2.0 * (top - 0.5)) * (1.0 - bottom),
                            2.0 * top * bottom)
    def apply_segmentation_effects(self, image_tensor, operating_mode, enable_dof, dof_strength, relighting_mode, relight_strength, mood_image, mood_strength, mood_background_replace):
        """
        Orchestrates segmentation-based effects, now passing the mood_background_replace toggle.
        """
        try:
            check_for_interruption()

            if not enable_dof and relighting_mode == "Disabled" and (mood_image is None or mood_strength <= 0):
                return image_tensor

            if self.remover is None:
                print("Initializing transparent-background remover (InspyreNet)...")
                try:
                    from transparent_background import Remover
                    self.remover = Remover(mode='base', jit=False)
                except Exception as e:
                    print(f"Failed to initialize transparent-background remover: {e}")
                    print("Segmentation-based effects (Relighting, DoF, Mood) will be skipped.")
                    return image_tensor

            device = image_tensor.device
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            image_bchw = image_tensor.permute(0, 3, 1, 2)
            
            img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            mask_pil = self.remover.process(pil_image, type='map')
            if mask_pil.mode != 'L': mask_pil = mask_pil.convert('L')
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            
            subject_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            
            if mood_image is not None and mood_strength > 0:
                print(f"Refiner: Applying Mood & Lighting Transfer (Mode: {'Replace' if mood_background_replace else 'Relight'})")
                image_bchw = self.apply_mood_and_lighting_transfer(image_bchw, subject_mask, mood_image, mood_strength, mood_background_replace)
                check_for_interruption()

            feather_amount = max(1.0, (min(h, w) / 512.0) * 2.0)
            kernel_size = int(feather_amount * 2) | 1
            compositing_mask = kornia.filters.gaussian_blur2d(subject_mask, (kernel_size, kernel_size), (feather_amount, feather_amount))
            
            sharp_foreground = image_bchw
            background = image_bchw

            if relighting_mode != "Disabled" and relight_strength > 0:
                if subject_mask.sum() > 0:
                    if relighting_mode == "Additive (Simple)":
                        sharp_foreground = self.apply_professional_relighting(image_bchw, compositing_mask, relight_strength * 0.4, device)
                    elif relighting_mode == "Corrective":
                        if operating_mode == "Anime":
                            sharp_foreground = self.apply_correction_anime(image_bchw, compositing_mask, relight_strength * 0.7, h, w)
                        else:
                            sharp_foreground = self.apply_correction_photo(image_bchw, compositing_mask, relight_strength)
                else:
                    print("Refiner: Relighting enabled, but no subject found in mask. Skipping.")
                check_for_interruption()
            
            if enable_dof and dof_strength > 0:
                background = self._apply_dof_pyramid(image_bchw, subject_mask, dof_strength, device)
            check_for_interruption()

            final_image = torch.lerp(background, sharp_foreground, compositing_mask)
            
            return final_image.permute(0, 2, 3, 1)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying segmentation effects: {e}")
            import traceback
            traceback.print_exc()
            return image_tensor
        
    def apply_correction_photo(self, image_bchw, subject_mask, strength):
        """
        Performs an advanced, local adaptive tone correction using CLAHE in the LAB
        color space. This method is highly effective at revealing detail in harsh
        highlights and deep shadows on a subject's face without affecting the overall
        image tone.
        """
        try:
            check_for_interruption()

            if strength <= 0:
                return image_bchw

            if not hasattr(kornia.enhance, 'equalize_clahe'):
                print("-------------------------------------------------------------------------------------------------")
                print("WARNING: kornia.enhance.equalize_clahe not found.")
                print("Your version of kornia may be outdated. Please update it.")
                print("Skipping Corrective (Auto) relighting.")
                print("-------------------------------------------------------------------------------------------------")
                return image_bchw

            x_min, y_min, x_max, y_max = self.get_mask_bounding_box(subject_mask)

            padding = 20
            h, w = image_bchw.shape[-2:]
            crop_x_min = max(0, x_min - padding)
            crop_y_min = max(0, y_min - padding)
            crop_x_max = min(w, x_max + padding)
            crop_y_max = min(h, y_max + padding)

            if crop_x_min >= crop_x_max or crop_y_min >= crop_y_max:
                    return image_bchw

            subject_crop_bchw = image_bchw[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            subject_crop_lab = kornia.color.rgb_to_lab(subject_crop_bchw)

            l_channel = subject_crop_lab[:, 0:1, :, :]

            clahe_clip_limit = 2.5
            corrected_l_channel = kornia.enhance.equalize_clahe(l_channel / 100.0, clip_limit=clahe_clip_limit) * 100.0

            corrected_lab_crop = torch.cat([corrected_l_channel, subject_crop_lab[:, 1:3, :, :]], dim=1)
            
            corrected_rgb_crop = kornia.color.lab_to_rgb(corrected_lab_crop)

            corrected_image = image_bchw.clone()
            corrected_image[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max] = corrected_rgb_crop
            
            blending_strength = pow(strength, 1.5)
            
            final_image = torch.lerp(image_bchw, corrected_image, blending_strength * subject_mask)

            print(f"Refiner (Corrective): Applied local adaptive correction (CLAHE) with strength {strength:.2f}")

            return torch.clamp(final_image, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in automatic light correction (CLAHE): {e}")
            import traceback
            traceback.print_exc()
            return image_bchw
       
    def apply_correction_anime(self, image_bchw, subject_mask, strength, h, w):
        """
        Anime-specific lighting correction that targets harsh highlights and crushed shadows
        within the segmented character while preserving the clean anime art style.
        """
        try:
            check_for_interruption()

            if strength <= 0:
                return image_bchw

            image_hsl = self.rgb_to_hsl_batch(image_bchw)
            
            character_lightness = self.extract_character_lightness(image_hsl, subject_mask)
            
            lighting_issues = self.analyze_character_lighting_issues(character_lightness)
            
            highlight_correction = self.create_highlight_correction_map(
                image_hsl, subject_mask, lighting_issues, strength
            )
            shadow_correction = self.create_shadow_correction_map(
                image_hsl, subject_mask, lighting_issues, strength
            )
            
            corrected_hsl = self.apply_anime_lighting_corrections(
                image_hsl, highlight_correction, shadow_correction, subject_mask
            )
            
            corrected_rgb = self.hsl_to_rgb_batch(corrected_hsl)
            
            return torch.clamp(corrected_rgb, 0.0, 1.0)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in anime lighting correction: {e}")
            import traceback
            traceback.print_exc()
            return image_bchw
    def rgb_to_hsl_batch(self, rgb_bchw):
        """Convert RGB to HSL color space for better anime color handling"""
        try:
            check_for_interruption()
            
            r, g, b = rgb_bchw[:, 0:1], rgb_bchw[:, 1:2], rgb_bchw[:, 2:3]
            
            max_val = torch.max(torch.max(r, g), b)
            min_val = torch.min(torch.min(r, g), b)
            diff = max_val - min_val
            
            lightness = (max_val + min_val) / 2.0
            
            saturation = torch.where(
                diff == 0,
                torch.zeros_like(diff),
                torch.where(
                    lightness < 0.5,
                    diff / (max_val + min_val + 1e-8),
                    diff / (2.0 - max_val - min_val + 1e-8)
                )
            )
            
            hue = torch.zeros_like(diff)
            
            red_mask = (max_val == r) & (diff != 0)
            hue = torch.where(red_mask, (g - b) / (diff + 1e-8), hue)
            
            green_mask = (max_val == g) & (diff != 0)
            hue = torch.where(green_mask, 2.0 + (b - r) / (diff + 1e-8), hue)
            
            blue_mask = (max_val == b) & (diff != 0)
            hue = torch.where(blue_mask, 4.0 + (r - g) / (diff + 1e-8), hue)
            
            hue = hue / 6.0
            hue = torch.where(hue < 0, hue + 1.0, hue)
            
            return torch.cat([hue, saturation, lightness], dim=1)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in RGB to HSL conversion: {e}")
            return torch.zeros_like(rgb_bchw)

    def hsl_to_rgb_batch(self, hsl_bchw):
        """Convert HSL back to RGB color space"""
        try:
            check_for_interruption()
            
            h, s, l = hsl_bchw[:, 0:1], hsl_bchw[:, 1:2], hsl_bchw[:, 2:3]
            
            def hue_to_rgb(p, q, t):
                t = torch.where(t < 0, t + 1, t)
                t = torch.where(t > 1, t - 1, t)
                
                result = torch.where(t < 1/6, p + (q - p) * 6 * t, p)
                result = torch.where((t >= 1/6) & (t < 1/2), q, result)
                result = torch.where((t >= 1/2) & (t < 2/3), p + (q - p) * (2/3 - t) * 6, result)
                
                return result
            
            rgb = torch.cat([l, l, l], dim=1)
            
            q = torch.where(l < 0.5, l * (1 + s), l + s - l * s)
            p = 2 * l - q
            
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
            
            chromatic_rgb = torch.cat([r, g, b], dim=1)
            
            mask = (s > 1e-8).expand_as(rgb)
            result = torch.where(mask, chromatic_rgb, rgb)
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in HSL to RGB conversion: {e}")
            return torch.zeros_like(hsl_bchw)
    def extract_character_lightness(self, image_hsl, subject_mask):
        """Extract only the character's lightness values for clean analysis"""
        try:
            check_for_interruption()
            
            lightness_channel = image_hsl[:, 2:3, :, :]
            character_lightness = lightness_channel * subject_mask
            
            valid_mask = subject_mask > 0.1
            valid_lightness = character_lightness[valid_mask]
            
            return valid_lightness
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error extracting character lightness: {e}")
            return torch.tensor([0.5], device=image_hsl.device)

    def analyze_character_lighting_issues(self, character_lightness):
        """Analyze the character's lighting with more practical thresholds for real anime images"""
        try:
            check_for_interruption()
            
            if character_lightness.numel() == 0:
                return self.get_default_lighting_analysis(character_lightness.device)
            
            char_mean = torch.mean(character_lightness)
            char_std = torch.std(character_lightness)
            char_min = torch.min(character_lightness)
            char_max = torch.max(character_lightness)
            
            highlight_threshold = 0.7
            harsh_highlights = torch.sum(character_lightness > highlight_threshold).float() / character_lightness.numel()
            
            shadow_threshold = 0.3
            crushed_shadows = torch.sum(character_lightness < shadow_threshold).float() / character_lightness.numel()
            
            is_overlit = char_mean > 0.6
            is_underlit = char_mean < 0.4
            has_poor_contrast = char_std < 0.2
            
            needs_highlight_correction = harsh_highlights > 0.02 or char_max > 0.8
            needs_shadow_correction = crushed_shadows > 0.02 or char_min < 0.25
            
            return {
                'mean_lightness': char_mean.item(),
                'lightness_std': char_std.item(),
                'min_lightness': char_min.item(),
                'max_lightness': char_max.item(),
                'harsh_highlights_ratio': max(harsh_highlights.item(), 0.1),
                'crushed_shadows_ratio': max(crushed_shadows.item(), 0.1),
                'is_overlit': is_overlit.item(),
                'is_underlit': is_underlit.item(),
                'has_poor_contrast': has_poor_contrast.item(),
                'needs_highlight_correction': needs_highlight_correction,
                'needs_shadow_correction': needs_shadow_correction
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error analyzing character lighting: {e}")
            return self.get_default_lighting_analysis(character_lightness.device)

    def get_default_lighting_analysis(self, device):
        """Return default lighting analysis when character analysis fails"""
        return {
            'mean_lightness': 0.5,
            'lightness_std': 0.2,
            'min_lightness': 0.1,
            'max_lightness': 0.9,
            'harsh_highlights_ratio': 0.0,
            'crushed_shadows_ratio': 0.0,
            'is_overlit': False,
            'is_underlit': False,
            'has_poor_contrast': False
        }
    def create_highlight_correction_map(self, image_hsl, subject_mask, lighting_issues, strength):
        """Create a more aggressive correction map for anime highlights"""
        try:
            check_for_interruption()
            
            lightness = image_hsl[:, 2:3, :, :]
            
            highlight_threshold = 0.6
            moderate_highlights = (lightness > highlight_threshold).float()
            strong_highlights = (lightness > 0.8).float()
            
            scale_factor = min(image_hsl.shape[-1], image_hsl.shape[-2]) / 1024.0
            blur_radius = max(6.0 * scale_factor, 3.0)
            kernel_size = int(blur_radius * 2) | 1
            
            moderate_correction = kornia.filters.gaussian_blur2d(
                moderate_highlights,
                (kernel_size, kernel_size),
                (blur_radius, blur_radius)
            )
            
            strong_correction = kornia.filters.gaussian_blur2d(
                strong_highlights,
                (kernel_size, kernel_size),
                (blur_radius, blur_radius)
            )
            
            base_moderate = 0.08 * strength
            base_strong = 0.15 * strength
            
            final_correction = (moderate_correction * base_moderate + 
                            strong_correction * base_strong) * subject_mask
            
            return torch.clamp(final_correction, 0.0, 0.4)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating highlight correction map: {e}")
            return torch.zeros_like(image_hsl[:, 2:3, :, :])

    def create_shadow_correction_map(self, image_hsl, subject_mask, lighting_issues, strength):
        """Create a more aggressive correction map for anime shadows"""
        try:
            check_for_interruption()
            
            lightness = image_hsl[:, 2:3, :, :]
            
            moderate_shadow_threshold = 0.45
            dark_shadow_threshold = 0.25
            
            moderate_shadows = (lightness < moderate_shadow_threshold).float()
            dark_shadows = (lightness < dark_shadow_threshold).float()
            
            scale_factor = min(image_hsl.shape[-1], image_hsl.shape[-2]) / 1024.0
            blur_radius = max(8.0 * scale_factor, 4.0)
            kernel_size = int(blur_radius * 2) | 1
            
            moderate_correction = kornia.filters.gaussian_blur2d(
                moderate_shadows,
                (kernel_size, kernel_size),
                (blur_radius, blur_radius)
            )
            
            dark_correction = kornia.filters.gaussian_blur2d(
                dark_shadows,
                (kernel_size, kernel_size),
                (blur_radius, blur_radius)
            )
            
            base_moderate = 0.06 * strength
            base_dark = 0.12 * strength
            
            final_correction = (moderate_correction * base_moderate + 
                            dark_correction * base_dark) * subject_mask
            
            return torch.clamp(final_correction, 0.0, 0.35)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating shadow correction map: {e}")
            return torch.zeros_like(image_hsl[:, 2:3, :, :])
    def apply_anime_lighting_corrections(self, image_hsl, highlight_correction, shadow_correction, subject_mask):
        """Apply lighting corrections to anime characters without position assumptions"""
        try:
            check_for_interruption()
            
            corrected_hsl = image_hsl.clone()
            lightness = corrected_hsl[:, 2:3, :, :]
            
            corrected_lightness = lightness - highlight_correction
            
            corrected_lightness = corrected_lightness + shadow_correction
            
            corrected_lightness = self.apply_character_tone_balancing(
                lightness, corrected_lightness, subject_mask
            )
            
            corrected_lightness = torch.clamp(corrected_lightness, 0.0, 1.0)
            
            corrected_hsl[:, 2:3, :, :] = corrected_lightness
            
            return corrected_hsl
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying anime lighting corrections: {e}")
            return image_hsl

    def apply_character_tone_balancing(self, original_lightness, corrected_lightness, subject_mask):
        """Apply gentle tone balancing and black point restoration to prevent washed out look"""
        try:
            check_for_interruption()
            
            enhanced = original_lightness + torch.sin(original_lightness * 3.14159) * 0.04
            enhanced = torch.clamp(enhanced, 0.0, 1.0)
            
            character_enhanced = torch.lerp(original_lightness, enhanced, subject_mask * 0.3)
            
            balanced_lightness = torch.lerp(corrected_lightness, character_enhanced, 0.5)
            
            restored_lightness = self.apply_black_point_restoration(
                original_lightness, balanced_lightness, subject_mask
            )
            
            return restored_lightness
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in character tone balancing: {e}")
            return corrected_lightness

    def apply_black_point_restoration(self, original_lightness, corrected_lightness, subject_mask):
        """Restore black point and contrast to prevent washed out appearance"""
        try:
            check_for_interruption()
            
            correction_amount = torch.abs(corrected_lightness - original_lightness) * subject_mask
            max_correction = torch.max(correction_amount)
            
            if max_correction > 0.02:
                gamma_strength = torch.clamp(max_correction * 2.0, 0.05, 0.2)
                gamma_value = 1.0 + gamma_strength
                
                gamma_corrected = torch.pow(corrected_lightness + 1e-8, gamma_value)
                
                correction_mask = torch.clamp(correction_amount * 10.0, 0.0, 1.0)
                final_lightness = torch.lerp(corrected_lightness, gamma_corrected, correction_mask * 0.6)
                
                return final_lightness
            else:
                return corrected_lightness
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in black point restoration: {e}")
            return corrected_lightness
        
        
    def apply_professional_relighting(self, image_bchw, subject_mask, strength, device):
        try:
            check_for_interruption()
            
            h, w = image_bchw.shape[-2:]
            scale_factor = min(h, w) / 1024.0
            
            mask_system = self.create_smooth_subject_mask(subject_mask, scale_factor)
            
            detail_preservation = self.create_detail_preservation_mask(image_bchw, subject_mask, scale_factor)
            
            key_light = self.create_key_light(h, w, device, scale_factor)
            fill_light = self.create_fill_light(h, w, device, scale_factor)
            rim_light = self.create_rim_light(h, w, device, scale_factor, mask_system['core_mask'])
            
            lighting_analysis = self.analyze_subject_lighting(image_bchw, subject_mask)
            
            relit_image = self.apply_adaptive_studio_lighting_smooth(
                image_bchw, mask_system, detail_preservation, 
                key_light, fill_light, rim_light, 
                lighting_analysis, strength
            )
            
            return relit_image
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in professional relighting: {e}")
            return image_bchw
    def apply_adaptive_studio_lighting_smooth(self, image_bchw, mask_system, detail_preservation,
                                            key_light, fill_light, rim_light, lighting_analysis, strength):
        try:
            check_for_interruption()

            if strength <= 0:
                return image_bchw

            fully_relit_image = image_bchw.clone()

            exposure_strength = 0.5
            key_strength = 0.35 * (0.8 if lighting_analysis['avg_brightness'] > 0.7 else 1.2)
            fill_strength = 0.2 + (lighting_analysis['shadow_severity'] * 0.25)
            rim_strength = 0.3
            color_strength = 0.6
            contrast_strength = 0.45

            fully_relit_image = self.apply_smart_exposure_correction_smooth(
                fully_relit_image, mask_system, detail_preservation, exposure_strength
            )
            
            fully_relit_image = self.apply_light_to_subject_smooth(
                fully_relit_image, mask_system, detail_preservation, key_light, key_strength, 'key'
            )
            fully_relit_image = self.apply_light_to_subject_smooth(
                fully_relit_image, mask_system, detail_preservation, fill_light, fill_strength, 'fill'
            )
            fully_relit_image = self.apply_light_to_subject_smooth(
                fully_relit_image, mask_system, detail_preservation, rim_light, rim_strength, 'rim'
            )

            if abs(lighting_analysis['color_temperature']) > 0.1:
                fully_relit_image = self.apply_color_temperature_correction_smooth(
                    fully_relit_image, mask_system, lighting_analysis['color_temperature'], color_strength
                )
            fully_relit_image = self.apply_subject_contrast_enhancement_smooth(
                fully_relit_image, mask_system, detail_preservation, contrast_strength
            )

            blending_strength = pow(strength, 1.5)

            relit_image = torch.lerp(image_bchw, fully_relit_image, blending_strength * mask_system['combined_mask'])
            
            if blending_strength > 0:
                gamma = 1.0 + (0.15 * blending_strength)
                
                darkened_image = torch.clamp(relit_image.pow(gamma), 0.0, 1.0)

                final_image = torch.lerp(relit_image, darkened_image, mask_system['combined_mask'])
            else:
                final_image = relit_image

            return final_image
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth adaptive studio lighting: {e}")
            return image_bchw
    def create_smooth_subject_mask(self, subject_mask, scale_factor):
        try:
            check_for_interruption()
            
            device = subject_mask.device
            
            edge_blur = max(3.0 * scale_factor, 2.0)
            edge_kernel_size = int(edge_blur * 2) | 1
            
            soft_mask = kornia.filters.gaussian_blur2d(
                subject_mask,
                (edge_kernel_size, edge_kernel_size),
                (edge_blur, edge_blur)
            )
            
            falloff_blur = max(15.0 * scale_factor, 8.0)
            falloff_kernel_size = int(falloff_blur * 2) | 1
            
            falloff_mask = kornia.filters.gaussian_blur2d(
                subject_mask,
                (falloff_kernel_size, falloff_kernel_size),
                (falloff_blur, falloff_blur)
            )
            
            smooth_mask = soft_mask * 0.7 + falloff_mask * 0.3
            
            edge_transition = torch.clamp(falloff_mask - soft_mask + 0.2, 0, 1)
            
            return {
                'core_mask': soft_mask,
                'falloff_mask': falloff_mask, 
                'transition_mask': edge_transition,
                'combined_mask': smooth_mask
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating smooth subject mask: {e}")
            return {
                'core_mask': subject_mask,
                'falloff_mask': subject_mask,
                'transition_mask': subject_mask,
                'combined_mask': subject_mask
            }
    def create_detail_preservation_mask(self, image_bchw, subject_mask, scale_factor):
        try:
            check_for_interruption()
            
            device = image_bchw.device
            
            gray = 0.299 * image_bchw[:, 0:1] + 0.587 * image_bchw[:, 1:2] + 0.114 * image_bchw[:, 2:3]
            
            detail_small = kornia.filters.sobel(gray)
            
            blur_medium = kornia.filters.gaussian_blur2d(gray, (5, 5), (1.5, 1.5))
            detail_medium = torch.abs(gray - blur_medium)
            
            blur_large = kornia.filters.gaussian_blur2d(gray, (9, 9), (3.0, 3.0))
            detail_large = torch.abs(gray - blur_large)
            
            combined_details = (detail_small * 0.4 + detail_medium * 0.4 + detail_large * 0.2)
            
            detail_strength = torch.tanh(combined_details * 8.0)
            
            shadow_areas = torch.exp(-(gray / 0.4)**2)
            detail_strength = detail_strength * shadow_areas
            
            smooth_radius = max(2.0 * scale_factor, 1.0)
            smooth_kernel_size = int(smooth_radius * 2) | 1
            
            detail_preservation = kornia.filters.gaussian_blur2d(
                detail_strength,
                (smooth_kernel_size, smooth_kernel_size),
                (smooth_radius, smooth_radius)
            )
            
            detail_preservation = detail_preservation * subject_mask
            
            return torch.clamp(detail_preservation, 0, 0.8)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error creating detail preservation mask: {e}")
            return torch.zeros_like(subject_mask)
    
    def create_key_light(self, h, w, device, scale_factor):
        light_x = 0.5
        light_y = 0.3
        
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        
        distance = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
        
        falloff = 1.0 / (1.0 + (distance / 0.8)**1.5)
        
        key_light = falloff
        
        blur_radius = max(25.0 * scale_factor, 8.0)
        kernel_size = int(blur_radius * 2) | 1
        key_light = kornia.filters.gaussian_blur2d(
            key_light.unsqueeze(0).unsqueeze(0),
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        ).squeeze()
        
        key_light = key_light * 0.15
        
        return key_light.unsqueeze(0).unsqueeze(0)

    def create_fill_light(self, h, w, device, scale_factor):
        light_x = 0.75
        light_y = 0.4
        
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        
        distance = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
        
        falloff = 1.0 / (1.0 + (distance / 0.6)**1.5)
        
        blur_radius = max(15.0 * scale_factor, 4.0)
        kernel_size = int(blur_radius * 2) | 1
        fill_light = kornia.filters.gaussian_blur2d(
            falloff.unsqueeze(0).unsqueeze(0),
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        ).squeeze()
        
        return fill_light.unsqueeze(0).unsqueeze(0) * 0.06

    def create_rim_light(self, h, w, device, scale_factor, subject_mask):
        
        blur_radius = max(2.0 * scale_factor, 1.0)
        kernel_size = int(blur_radius * 2) | 1
        
        soft_mask = kornia.filters.gaussian_blur2d(
            subject_mask,
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        )
        
        edge_kernel = torch.tensor([[-0.5, -0.5, -0.5], [-0.5, 4, -0.5], [-0.5, -0.5, -0.5]], 
                                dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        edges = F.conv2d(soft_mask, edge_kernel, padding=1)
        edges = torch.clamp(torch.abs(edges), 0, 1)
        
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        
        rim_direction = torch.clamp(x_coords * 0.3 + (1.0 - y_coords) * 0.2 + 0.5, 0, 1)
        
        rim_light = edges * rim_direction
        
        blur_radius = max(8.0 * scale_factor, 3.0)
        kernel_size = int(blur_radius * 2) | 1
        rim_light = kornia.filters.gaussian_blur2d(
            rim_light,
            (kernel_size, kernel_size),
            (blur_radius, blur_radius)
        )
        
        return rim_light * 0.05
    def analyze_subject_lighting(self, image_bchw, subject_mask):
        try:
            check_for_interruption()
            
            subject_pixels = image_bchw * subject_mask
            valid_pixels = subject_pixels[subject_mask.bool().expand_as(subject_pixels)]
            
            if valid_pixels.numel() == 0:
                return {
                    'avg_brightness': 0.5,
                    'contrast_ratio': 1.0,
                    'is_underlit': False,
                    'shadow_severity': 0.0,
                    'color_temperature': 0.0
                }
            
            valid_pixels = valid_pixels.view(-1, 3)
            
            luminance = 0.299 * valid_pixels[:, 0] + 0.587 * valid_pixels[:, 1] + 0.114 * valid_pixels[:, 2]
            
            avg_brightness = torch.mean(luminance)
            brightness_std = torch.std(luminance)
            
            is_underlit = avg_brightness < 0.4
            
            shadow_pixels = (luminance < 0.2).float()
            shadow_severity = torch.mean(shadow_pixels)
            
            avg_red = torch.mean(valid_pixels[:, 0])
            avg_blue = torch.mean(valid_pixels[:, 2])
            color_temperature = (avg_red - avg_blue).clamp(-0.3, 0.3)
            
            contrast_ratio = brightness_std / (avg_brightness + 0.001)
            
            return {
                'avg_brightness': avg_brightness.item(),
                'contrast_ratio': contrast_ratio.item(),
                'is_underlit': is_underlit.item(),
                'shadow_severity': shadow_severity.item(),
                'color_temperature': color_temperature.item()
            }
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in lighting analysis: {e}")
            return {
                'avg_brightness': 0.5,
                'contrast_ratio': 1.0,
                'is_underlit': False,
                'shadow_severity': 0.0,
                'color_temperature': 0.0
            }

    
    def apply_smart_exposure_correction_smooth(self, image_bchw, mask_system, detail_preservation, strength):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
                
            result = image_bchw.clone()
            
            luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            
            shadow_mask = torch.exp(-(luminance / 0.3)**2)
            midtone_mask = torch.exp(-((luminance - 0.5) / 0.3)**2)
            highlight_mask = torch.exp(-((luminance - 0.8) / 0.2)**2)
            
            detail_reduction = 1.0 - (detail_preservation * 0.6)
            
            shadow_lift = strength * 0.05 * shadow_mask * detail_reduction
            midtone_boost = strength * 0.02 * midtone_mask * detail_reduction
            highlight_protection = -strength * 0.01 * highlight_mask
            
            core_adjustment = (shadow_lift + midtone_boost + highlight_protection) * mask_system['core_mask']
            falloff_adjustment = (shadow_lift + midtone_boost + highlight_protection) * 0.6 * mask_system['falloff_mask']
            transition_adjustment = (shadow_lift + midtone_boost + highlight_protection) * 0.3 * mask_system['transition_mask']
            
            total_adjustment = core_adjustment + falloff_adjustment + transition_adjustment
            
            total_adjustment = torch.clamp(total_adjustment, -0.1, 0.1)
            
            for c in range(3):
                channel = result[:, c:c+1, :, :]
                adjusted = channel + total_adjustment
                result[:, c:c+1, :, :] = torch.clamp(adjusted, 0, 1)
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth smart exposure correction: {e}")
            return image_bchw
    def apply_subject_contrast_enhancement_smooth(self, image_bchw, mask_system, detail_preservation, strength):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
                
            result = image_bchw.clone()
            
            luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            
            detail_reduction = 1.0 - (detail_preservation * 0.8)
            
            enhanced_luminance = luminance + torch.sin(luminance * 3.14159) * strength * 0.1 * detail_reduction
            enhanced_luminance = torch.clamp(enhanced_luminance, 0, 1)
            
            luminance_ratio = enhanced_luminance / (luminance + 1e-8)
            luminance_ratio = torch.clamp(luminance_ratio, 0.7, 1.5)
            
            core_ratio = luminance_ratio * mask_system['core_mask']
            falloff_ratio = torch.lerp(torch.ones_like(luminance_ratio), luminance_ratio, 0.6) * mask_system['falloff_mask']
            transition_ratio = torch.lerp(torch.ones_like(luminance_ratio), luminance_ratio, 0.3) * mask_system['transition_mask']
            
            background_ratio = torch.ones_like(luminance_ratio) * (1.0 - torch.clamp(
                mask_system['core_mask'] + mask_system['falloff_mask'] + mask_system['transition_mask'], 0, 1
            ))
            
            final_ratio = core_ratio + falloff_ratio + transition_ratio + background_ratio
            
            for c in range(3):
                channel = result[:, c:c+1, :, :]
                enhanced_channel = channel * final_ratio
                result[:, c:c+1, :, :] = torch.clamp(enhanced_channel, 0, 1)
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth subject contrast enhancement: {e}")
            return image_bchw
    def apply_light_to_subject_smooth(self, image_bchw, mask_system, detail_preservation, light_map, strength, light_type):
        try:
            check_for_interruption()
            
            if strength <= 0:
                return image_bchw
            
            device = image_bchw.device
            
            if light_map.shape[-2:] != image_bchw.shape[-2:]:
                light_map = F.interpolate(light_map, size=image_bchw.shape[-2:], mode='bilinear', align_corners=False)
            
            result = image_bchw.clone()
            
            if light_type == 'key':
                warm_r, warm_g, warm_b = 1.0, 0.98, 0.95
                
                luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
                shadow_mask = torch.exp(-(luminance / 0.4)**2)
                
                detail_reduction = 1.0 - (detail_preservation * 0.3)
                
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    light_color = [warm_r, warm_g, warm_b][c]
                    
                    core_strength = strength * shadow_mask * detail_reduction * mask_system['core_mask']
                    falloff_strength = strength * 0.4 * shadow_mask * mask_system['falloff_mask']
                    transition_strength = strength * 0.2 * shadow_mask * mask_system['transition_mask']
                    
                    total_strength = torch.clamp(core_strength + falloff_strength + transition_strength, 0, strength)
                    
                    light_effect = light_map * total_strength * light_color * 0.8 
                    blended_channel = 1.0 - (1.0 - channel) * (1.0 - light_effect)
                    result[:, c:c+1, :, :] = torch.clamp(blended_channel, 0, 1)

            elif light_type == 'fill':
                luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
                shadow_mask = torch.exp(-(luminance / 0.25)**2)
                
                detail_reduction = 1.0 - (detail_preservation * 0.5)
                fill_colors = [0.98, 1.0, 1.02]
                
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    fill_color = fill_colors[c]
                    
                    core_fill = strength * 0.3 * shadow_mask * detail_reduction * mask_system['core_mask']
                    falloff_fill = strength * 0.2 * shadow_mask * mask_system['falloff_mask']
                    
                    combined_fill_effect = light_map * (core_fill + falloff_fill) * fill_color
                    lifted = 1.0 - (1.0 - channel) * (1.0 - combined_fill_effect)
                    result[:, c:c+1, :, :] = torch.clamp(lifted, 0, 1)
            
            elif light_type == 'rim':
                for c in range(3):
                    channel = result[:, c:c+1, :, :]
                    
                    rim_effect = light_map * strength * 0.6 * mask_system['core_mask']
                    
                    highlighted = 1.0 - (1.0 - channel) * (1.0 - rim_effect)
                    
                    rim_lerp_mask = torch.clamp(rim_effect * 10.0, 0, 1)
                    result[:, c:c+1, :, :] = torch.lerp(channel, highlighted, rim_lerp_mask)
            
            return torch.clamp(result, 0, 1)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error applying {light_type} light with smooth blending: {e}")
            return image_bchw
    def apply_color_temperature_correction_smooth(self, image_bchw, mask_system, temperature_bias, strength):
        try:
            check_for_interruption()
            
            if abs(temperature_bias) < 0.05 or strength <= 0:
                return image_bchw
            
            result = image_bchw.clone()
            
            if temperature_bias > 0:
                red_adjust = -temperature_bias * strength
                blue_adjust = temperature_bias * strength * 0.8
            else:
                red_adjust = -temperature_bias * strength * 0.8
                blue_adjust = temperature_bias * strength
            
            core_strength = mask_system['core_mask']
            falloff_strength = mask_system['falloff_mask'] * 0.6
            transition_strength = mask_system['transition_mask'] * 0.3
            
            total_mask = core_strength + falloff_strength + transition_strength
            
            result[:, 0:1, :, :] = torch.clamp(
                result[:, 0:1, :, :] + red_adjust * total_mask, 0, 1
            )
            
            result[:, 2:3, :, :] = torch.clamp(
                result[:, 2:3, :, :] + blue_adjust * total_mask, 0, 1
            )
            
            return result
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smooth color temperature correction: {e}")
            return image_bchw
    
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
   
    def apply_latent_refinement(self, latent_samples, strength):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return latent_samples
            

            sharpen_strength = pow(strength, 0.75)
            
            balance_strength = pow(strength, 2.0) * 0.5

            noise_enhancement_strength = pow(strength, 1.5) * 0.6

            
            refined_samples = latent_samples
            
            refined_samples = self.balance_latent_channels(refined_samples, balance_strength)
            check_for_interruption()
            
            refined_samples = self.enhance_latent_noise_structure(refined_samples, noise_enhancement_strength)
            check_for_interruption()
            
            refined_samples = self.sharpen_latent_detail(refined_samples, sharpen_strength)
            check_for_interruption()
            
            return refined_samples
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in apply_latent_refinement: {e}")
            return latent_samples
    def balance_latent_channels(self, latent_samples, strength=0.3):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return latent_samples
                
            device = latent_samples.device
            dtype = latent_samples.dtype
            
            balanced_samples = latent_samples.clone()
            
            for i in range(4):
                channel = balanced_samples[:, i:i+1, :, :]
                
                current_mean = torch.mean(channel)
                current_std = torch.std(channel)
                
                mean_threshold = 2.0
                if torch.abs(current_mean) > mean_threshold:
                    adjustment = (torch.abs(current_mean) - mean_threshold) * torch.sign(current_mean)
                    mean_adjustment = -adjustment * strength * 0.1
                    channel = channel + mean_adjustment
                
                if current_std < 0.1:
                    noise_boost = torch.randn_like(channel) * strength * 0.02
                    channel = channel + noise_boost
                elif current_std > 2.5:
                    smoothing_factor = 1.0 - (strength * 0.15)
                    channel = (channel - current_mean) * smoothing_factor + current_mean
                
                balanced_samples[:, i:i+1, :, :] = channel
            
            return balanced_samples
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in balance_latent_channels: {e}")
            return latent_samples
    def analyze_latent_noise_quality(self, latent_samples):
        try:
            check_for_interruption()
            
            device = latent_samples.device
            dtype = latent_samples.dtype
            
            channel_stats = {}
            for i in range(4):
                channel = latent_samples[:, i:i+1, :, :]
                
                mean = torch.mean(channel)
                std = torch.std(channel)
                
                grad_x = torch.abs(channel[:, :, :, 1:] - channel[:, :, :, :-1])
                grad_y = torch.abs(channel[:, :, 1:, :] - channel[:, :, :-1, :])
                
                high_freq_content = (torch.mean(grad_x) + torch.mean(grad_y)) / 2.0
                
                local_variance = F.avg_pool2d(channel.pow(2), 8, stride=4) - F.avg_pool2d(channel, 8, stride=4).pow(2)
                dead_zone_ratio = torch.mean((local_variance < 0.001).float())
                
                channel_stats[i] = {
                    'mean': mean,
                    'std': std,
                    'high_freq': high_freq_content,
                    'dead_zones': dead_zone_ratio
                }
            
            return channel_stats
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in analyze_latent_noise_quality: {e}")
            return {}
    def enhance_latent_noise_structure(self, latent_samples, strength=0.3):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return latent_samples
                
            device = latent_samples.device
            dtype = latent_samples.dtype
            
            noise_stats = self.analyze_latent_noise_quality(latent_samples)
            enhanced_samples = latent_samples.clone()
            
            for channel_idx in range(4):
                channel = enhanced_samples[:, channel_idx:channel_idx+1, :, :]
                stats = noise_stats.get(channel_idx, {})
                
                if stats.get('dead_zones', 0) > 0.3:
                    dead_zone_mask = self.create_dead_zone_mask(channel)
                    structured_noise = self.generate_structured_noise(channel.shape, device, dtype)
                    
                    noise_amount = strength * 0.005 * dead_zone_mask
                    channel = channel + structured_noise * noise_amount
                
                if stats.get('high_freq', 0) < 0.05:
                    detail_enhancement = self.enhance_latent_detail(channel, strength * 0.2)
                    channel = channel + detail_enhancement
                
                enhanced_samples[:, channel_idx:channel_idx+1, :, :] = channel
            
            return enhanced_samples
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in enhance_latent_noise_structure: {e}")
            return latent_samples
    def sharpen_latent_detail(self, latent_samples, strength=0.3):
        try:
            check_for_interruption()
            
            if strength == 0.0:
                return latent_samples
                
            device = latent_samples.device
            dtype = latent_samples.dtype
            
            sharpened_samples = latent_samples.clone()
            
            for i in range(4):
                channel = sharpened_samples[:, i:i+1, :, :]
                original_mean = torch.mean(channel)
                
                blur_small = kornia.filters.gaussian_blur2d(channel, (3, 3), (0.8, 0.8))
                blur_medium = kornia.filters.gaussian_blur2d(channel, (5, 5), (1.5, 1.5))
                
                detail_fine = channel - blur_small
                detail_medium = channel - blur_medium
                combined_detail = (detail_fine * 0.6 + detail_medium * 0.4)
                
                max_detail_value = 1.5
                combined_detail = torch.clamp(combined_detail, -max_detail_value, max_detail_value)
                
                detail_magnitude = torch.abs(combined_detail)
                adaptive_mask = torch.tanh(detail_magnitude * 10.0)
                
                sharpening_amount = strength * 0.5 * adaptive_mask
                enhanced_channel = channel + combined_detail * sharpening_amount
                
                current_mean = torch.mean(enhanced_channel)
                mean_correction = original_mean - current_mean
                enhanced_channel = enhanced_channel + mean_correction
                
                sharpened_samples[:, i:i+1, :, :] = enhanced_channel
            
            return sharpened_samples
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in sharpen_latent_detail: {e}")
            return latent_samples
    def create_dead_zone_mask(self, channel):
        try:
            check_for_interruption()
            
            local_variance = F.avg_pool2d(channel.pow(2), 8, stride=1, padding=4) - F.avg_pool2d(channel, 8, stride=1, padding=4).pow(2)
            
            dead_zone_mask = (local_variance < 0.001).float()
            
            kernel_size = 5
            smooth_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=channel.dtype, device=channel.device) / (kernel_size * kernel_size)
            dead_zone_mask = F.conv2d(dead_zone_mask, smooth_kernel, padding=kernel_size//2)
            
            return dead_zone_mask
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in create_dead_zone_mask: {e}")
            return torch.zeros_like(channel)

    def generate_structured_noise(self, shape, device, dtype):
        try:
            check_for_interruption()
            
            noise_base = torch.randn(shape, device=device, dtype=dtype)
            
            h, w = shape[-2], shape[-1]
            
            if h > 16 and w > 16:
                noise_low = F.interpolate(
                    torch.randn((shape[0], shape[1], h//4, w//4), device=device, dtype=dtype),
                    size=(h, w), mode='bilinear', align_corners=False
                )
                noise_base = noise_base * 0.7 + noise_low * 0.3
            
            if h > 8 and w > 8:
                noise_base = kornia.filters.gaussian_blur2d(noise_base, (3, 3), (0.5, 0.5))
            
            return noise_base
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in generate_structured_noise: {e}")
            return torch.randn(shape, device=device, dtype=dtype)

    def enhance_latent_detail(self, channel, strength):
        try:
            check_for_interruption()
            
            grad_x = channel[:, :, :, 1:] - channel[:, :, :, :-1]
            grad_y = channel[:, :, 1:, :] - channel[:, :, :-1, :]
            
            grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
            
            enhanced_grad_x = torch.tanh(grad_x * 3.0) * strength * 0.1
            enhanced_grad_y = torch.tanh(grad_y * 3.0) * strength * 0.1
            
            detail_enhancement = (enhanced_grad_x + enhanced_grad_y) * 0.5
            
            return detail_enhancement
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in enhance_latent_detail: {e}")
            return torch.zeros_like(channel)
