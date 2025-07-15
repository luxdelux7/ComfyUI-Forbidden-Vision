import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import nodes
import cv2
from .utils import check_for_interruption

class LatentAIUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = folder_paths.get_filename_list("upscale_models")
        if not upscaler_models:
            upscaler_models = ["None Found"]

        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "upscale_model": (upscaler_models, ),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 8.0, "step": 0.05}),
                "enable_smart_sharpen": ("BOOLEAN", {"default": False}),
                "sharpening_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "use_tiled_vae": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("upscaled_latent", "upscaled_image")
    FUNCTION = "upscale_and_process"
    CATEGORY = "Forbidden Vision/Latent"

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
            self.upscaler_model = None
            self.upscaler_model_name = None
            return None
        
    def upscale_and_process(self, latent, vae, upscale_model, upscale_factor,
                            enable_smart_sharpen, sharpening_strength,
                            use_tiled_vae, tile_size):
        try:
            check_for_interruption()
            
            device = model_management.get_torch_device()
            decoded_image = vae.decode(latent["samples"].to(device)).cpu()

            check_for_interruption()

            loaded_model = self.load_upscaler_model(upscale_model)
            if loaded_model is None:
                raise Exception(f"Failed to load upscaler model: {upscale_model}")

            check_for_interruption()

            ImageUpscalerClass = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']
            upscaler_node = ImageUpscalerClass()
            ai_upscaled_image = upscaler_node.upscale(upscale_model=loaded_model, image=decoded_image)[0]
            
            check_for_interruption()
            
            # Get original and target dimensions
            h, w = decoded_image.shape[1], decoded_image.shape[2]
            target_h, target_w = int(h * upscale_factor), int(w * upscale_factor)
            
            final_image = ai_upscaled_image
            # If the AI upscaled image is not already at the target size, resize it.
            if ai_upscaled_image.shape[1] != target_h or ai_upscaled_image.shape[2] != target_w:
                # Use antialias=True for high-quality downsampling that prevents aliasing artifacts.
                final_image = F.interpolate(ai_upscaled_image.movedim(-1, 1), 
                                            size=(target_h, target_w), 
                                            mode='bicubic', 
                                            align_corners=False,
                                            antialias=True) # THIS IS THE KEY ADDITION
                final_image = final_image.movedim(1, -1)

            check_for_interruption()

            if enable_smart_sharpen and sharpening_strength > 0:
                final_image = self.apply_smart_sharpen(final_image, sharpening_strength)

            check_for_interruption()

            # Ensure final image is on the correct device before encoding
            final_image = final_image.to(device)

            if use_tiled_vae:
                VAEEncodeTiledClass = nodes.NODE_CLASS_MAPPINGS['VAEEncodeTiled']
                encoder = VAEEncodeTiledClass()
                vae_overlap = 64 # A common default
                upscaled_latent = encoder.encode(vae, final_image, tile_size, vae_overlap)[0]
            else:
                VAEEncodeClass = nodes.NODE_CLASS_MAPPINGS['VAEEncode']
                encoder = VAEEncodeClass()
                upscaled_latent = encoder.encode(vae, final_image)[0]

            if not isinstance(upscaled_latent, dict):
                upscaled_latent = {"samples": upscaled_latent}

            return (upscaled_latent, final_image.cpu())

        except model_management.InterruptProcessingException:
            print("Latent AI Upscaler cancelled by user")
            raise
        except Exception as e:
            print(f"Error in Latent AI Upscaler: {e}")
            import traceback
            traceback.print_exc()
            # Return latent in the original device state to prevent downstream errors
            return ({"samples": latent["samples"]}, torch.zeros((1, 64, 64, 3), dtype=torch.float32))
   
    def get_color_matrices(self, device, dtype):
        """Cache color conversion matrices."""
        if not hasattr(self, '_color_cache'):
            self._color_cache = {}
        
        cache_key = (device, dtype)
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]
        
        rgb_to_ycbcr_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                            [-0.168736, -0.331264, 0.5],
                                            [0.5, -0.418688, -0.081312]], dtype=dtype, device=device).t()
        
        ycbcr_to_rgb_matrix = torch.tensor([[1., 0., 1.402],
                                            [1., -0.344136, -0.714136],
                                            [1., 1.772, 0.]], dtype=dtype, device=device).t()
        
        bias = torch.tensor([0., 0.5, 0.5], dtype=dtype, device=device)
        
        self._color_cache[cache_key] = (rgb_to_ycbcr_matrix, ycbcr_to_rgb_matrix, bias)
        return rgb_to_ycbcr_matrix, ycbcr_to_rgb_matrix, bias
    
    def apply_smart_sharpen(self, image_tensor, strength):
        """Optimized smart sharpening with reduced CPU-GPU transfers."""
        try:
            device = image_tensor.device
            image_ycbcr = self.rgb_to_ycbcr(image_tensor)
            luma = image_ycbcr[..., 0:1]
            luma_bchw = luma.permute(0, 3, 1, 2)
            
            sobel_x_kernel, sobel_y_kernel = self.get_sobel_kernels(device, luma_bchw.dtype)
            sobel_x = F.conv2d(luma_bchw, sobel_x_kernel, padding=1)
            sobel_y = F.conv2d(luma_bchw, sobel_y_kernel, padding=1)
            edge_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
            
            luma_weight = 1.0 - torch.abs(luma_bchw - 0.5) * 1.2
            luma_weight = torch.clamp(luma_weight, 0.2, 1.0)
            
            contrast_mask = torch.clamp(edge_magnitude / (edge_magnitude.max() + 1e-6), 0.0, 1.0)
            contrast_mask = contrast_mask * luma_weight
            
            kernel_sizes = [3, 5, 7]
            sigmas = [0.8, 1.5, 2.5]
            weights = [0.4, 0.35, 0.25]
            
            combined_detail = torch.zeros_like(luma_bchw)
            
            for kernel_size, sigma, weight in zip(kernel_sizes, sigmas, weights):
                kernel = self.create_gaussian_kernel(kernel_size, sigma, device, luma_bchw.dtype)
                blurred = F.conv2d(luma_bchw, kernel, padding=kernel_size//2)
                detail = (luma_bchw - blurred) * weight
                combined_detail += detail
            
            detail_magnitude = torch.abs(combined_detail)
            noise_threshold = 0.02
            noise_mask = (detail_magnitude > noise_threshold).float()
            
            final_mask = contrast_mask * noise_mask
            sharpened_luma_bchw = luma_bchw + combined_detail * strength * final_mask
            
            final_ycbcr = torch.cat((sharpened_luma_bchw.permute(0, 2, 3, 1), image_ycbcr[..., 1:3]), dim=-1)
            final_rgb = self.ycbcr_to_rgb(final_ycbcr)
            
            return torch.clamp(final_rgb, 0.0, 1.0)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in smart sharpen: {e}")
            return image_tensor

    def create_gaussian_kernel(self, kernel_size, sigma, device, dtype):
        """Create Gaussian kernel on GPU to avoid CPU operations."""
        if not hasattr(self, '_kernel_cache'):
            self._kernel_cache = {}
        
        cache_key = (kernel_size, sigma, device, dtype)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d[None, None, :, :]
        
        self._kernel_cache[cache_key] = kernel
        if len(self._kernel_cache) > 10:
            oldest_key = next(iter(self._kernel_cache))
            del self._kernel_cache[oldest_key]
        
        return kernel

    def get_sobel_kernels(self, device, dtype):
        """Cache Sobel kernels to avoid recreation."""
        if not hasattr(self, '_sobel_cache'):
            self._sobel_cache = {}
        
        cache_key = (device, dtype)
        if cache_key in self._sobel_cache:
            return self._sobel_cache[cache_key]
        
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=dtype, device=device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=dtype, device=device)
        
        self._sobel_cache[cache_key] = (sobel_x, sobel_y)
        return sobel_x, sobel_y

    def rgb_to_ycbcr(self, image_rgb):
        """Convert RGB to YCbCr color space."""
        rgb_to_ycbcr_matrix, _, bias = self.get_color_matrices(image_rgb.device, image_rgb.dtype)
        return torch.matmul(image_rgb, rgb_to_ycbcr_matrix) + bias

    def ycbcr_to_rgb(self, image_ycbcr):
        """Convert YCbCr to RGB color space."""
        _, ycbcr_to_rgb_matrix, bias = self.get_color_matrices(image_ycbcr.device, image_ycbcr.dtype)
        return torch.matmul(image_ycbcr - bias, ycbcr_to_rgb_matrix)