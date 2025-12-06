import torch
import torch.nn.functional as F
import numpy as np
import nodes
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import comfy.samplers
import comfy.sample
import latent_preview
import cv2
import kornia
from .mask_processor import ForbiddenVisionMaskProcessor
from .utils import check_for_interruption, get_ordered_upscaler_model_list, ensure_model_directories, clean_model_name

class ForbiddenVisionInpaintLite:

    @classmethod
    def INPUT_TYPES(s):
        upscaler_models = get_ordered_upscaler_model_list()
        
        default_upscaler = "Fast 4x (Lanczos)"
        if "Fast 4x (Lanczos)" not in upscaler_models and upscaler_models:
            default_upscaler = upscaler_models[0]
        
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "mask": ("MASK",),

                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "denoise": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                "processing_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "crop_padding": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Padding added to the inpaint region."}),
                
                "manual_rotation": (["None", "90° CW", "90° CCW", "180°"], {"default": "None", "tooltip": "Manually rotate face crop before processing"}),

                "enable_pre_upscale": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "upscaler_model": (upscaler_models, {"default": default_upscaler}),
                "mask_expansion": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "sampling_mask_blur_size": ("INT", {"default": 21, "min": 1, "max": 101, "step": 2}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.1}),

                "blend_softness": ("INT", {"default": 12, "min": 0, "max": 200, "step": 1}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_differential_diffusion": ("BOOLEAN", {"default": True}),
                "bypass_cropping": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Skip cropping and perform full-image inpainting"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "process_inpaint"
    CATEGORY = "Forbidden Vision"
   
    def __init__(self):
        ensure_model_directories()
        self.upscaler_model = None
        self.upscaler_model_name = None
        self.mask_processor = ForbiddenVisionMaskProcessor()
    
    def apply_manual_rotation(self, image_np, rotation_option):
        if rotation_option == "None":
            return image_np
        elif rotation_option == "90° CW":
            return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_option == "90° CCW":
            return cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_option == "180°":
            return cv2.rotate(image_np, cv2.ROTATE_180)
        return image_np

    def reverse_manual_rotation(self, image_np, rotation_option):
        if rotation_option == "None":
            return image_np
        elif rotation_option == "90° CW":
            return cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_option == "90° CCW":
            return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_option == "180°":
            return cv2.rotate(image_np, cv2.ROTATE_180)
        return image_np

    def differential_diffusion_function(self, sigma, denoise_mask, extra_options):
        try:
            model = extra_options["model"]
            step_sigmas = extra_options["sigmas"]

            inner_model = getattr(model, "inner_model", model)
            sampling = inner_model.model_sampling

            sigma_to = sampling.sigma_min
            if step_sigmas[-1] > sigma_to:
                sigma_to = step_sigmas[-1]
            sigma_from = step_sigmas[0]

            ts_from = sampling.timestep(sigma_from)
            ts_to = sampling.timestep(sigma_to)
            current_ts = sampling.timestep(sigma[0])

            threshold = (current_ts - ts_to) / (ts_from - ts_to)
            binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

            strength = 1.0
            if strength and strength < 1.0:
                blended_mask = strength * binary_mask + (1.0 - strength) * denoise_mask
                return blended_mask
            else:
                return binary_mask

        except Exception as e:
            print(f"[ForbiddenVision] Differential diffusion mask error: {e}")
            return denoise_mask
        
    def load_upscaler_model(self, model_name):
        clean_model_name_val = clean_model_name(model_name)
        
        if clean_model_name_val in ["Fast 4x (Bicubic AA)", "Fast 4x (Lanczos)", "Fast 2x (Bicubic AA)", "Fast 2x (Lanczos)"]:
            self.upscaler_model = None
            self.upscaler_model_name = clean_model_name_val
            return True

        if self.upscaler_model is not None and self.upscaler_model_name == clean_model_name_val:
            return True

        try:
            model_path = folder_paths.get_full_path("upscale_models", clean_model_name_val)
            if model_path is None:
                print(f"Upscaler model '{clean_model_name_val}' not found.")
                return False
            
            UpscalerLoaderClass = nodes.NODE_CLASS_MAPPINGS['UpscaleModelLoader']
            upscaler_loader = UpscalerLoaderClass()
            
            self.upscaler_model = upscaler_loader.load_model(clean_model_name_val)[0]
            self.upscaler_model_name = clean_model_name_val
            
            return True
            
        except model_management.InterruptProcessingException:
            raise
        except KeyError:
            print(f"UpscaleModelLoader not found in NODE_CLASS_MAPPINGS")
            return False
        except Exception as e:
            print(f"Error loading upscaler model '{clean_model_name_val}': {e}")
            return False

    def upscale_image_wrapper(self, image_np_uint8):
        return self.upscale_image(image_np_uint8, self.upscaler_model_name)

    def upscale_image(self, image_np_uint8, upscaler_model_name):
        try:
            check_for_interruption()
            clean_name = clean_model_name(upscaler_model_name)
            
            if clean_name == "Fast 4x (Bicubic AA)":
                return self.fast_upscale_bicubic(image_np_uint8, scale=4)
            elif clean_name == "Fast 4x (Lanczos)":
                return self.fast_upscale_lanczos(image_np_uint8, scale=4)
            elif clean_name == "Fast 2x (Bicubic AA)":
                return self.fast_upscale_bicubic(image_np_uint8, scale=2)
            elif clean_name == "Fast 2x (Lanczos)":
                return self.fast_upscale_lanczos(image_np_uint8, scale=2)
            
            if self.upscaler_model is None or self.upscaler_model_name != clean_name:
                self.load_upscaler_model(clean_name)
            
            if self.upscaler_model is None:
                return image_np_uint8

            ImageUpscaleWithModelClass = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']
            image_upscaler = ImageUpscaleWithModelClass()
            
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0).unsqueeze(0)
            upscaled_tensor = image_upscaler.upscale(self.upscaler_model, image_tensor)[0]
            
            upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy()
            upscaled_np_uint8 = (np.clip(upscaled_np, 0, 1) * 255.0).round().astype(np.uint8)
            
            return upscaled_np_uint8
            
        except Exception as e:
            print(f"Error in upscaling: {e}. Returning original image.")
            return image_np_uint8

    def fast_upscale_bicubic(self, image_np_uint8, scale=4):
        try:
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0).unsqueeze(0)
            with torch.no_grad():
                upscaled_tensor = F.interpolate(
                    image_tensor.permute(0, 3, 1, 2),
                    scale_factor=scale,
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)
            
            upscaled_np = upscaled_tensor.squeeze(0).cpu().numpy()
            upscaled_np_uint8 = (np.clip(upscaled_np, 0, 1) * 255.0).round().astype(np.uint8)
            return self.clean_interpolation_edges(upscaled_np_uint8)
        except Exception:
            return image_np_uint8

    def fast_upscale_lanczos(self, image_np_uint8, scale=4):
        try:
            h, w = image_np_uint8.shape[:2]
            new_h, new_w = h * scale, w * scale
            upscaled_np_uint8 = cv2.resize(image_np_uint8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            return self.clean_interpolation_edges(upscaled_np_uint8)
        except Exception:
            return image_np_uint8
    
    def clean_interpolation_edges(self, image_np_uint8):
        try:
            image_tensor = torch.from_numpy(image_np_uint8.astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            edge_size = max(8, min(h, w) // 32)
            
            mask = torch.ones_like(image_tensor[:, :1, :, :])
            mask[:, :, :edge_size, :] = torch.linspace(0, 1, edge_size).view(1, 1, edge_size, 1)
            mask[:, :, h-edge_size:, :] = torch.linspace(1, 0, edge_size).view(1, 1, edge_size, 1)
            mask[:, :, :, :edge_size] = torch.minimum(mask[:, :, :, :edge_size], torch.linspace(0, 1, edge_size).view(1, 1, 1, edge_size))
            mask[:, :, :, w-edge_size:] = torch.minimum(mask[:, :, :, w-edge_size:], torch.linspace(1, 0, edge_size).view(1, 1, 1, edge_size))
            
            kernel_size = max(3, edge_size // 2)
            if kernel_size % 2 == 0: kernel_size += 1
            sigma = kernel_size / 3.0
            
            blurred = kornia.filters.gaussian_blur2d(image_tensor, (kernel_size, kernel_size), (sigma, sigma))
            cleaned = image_tensor * mask + blurred * (1 - mask)
            
            cleaned_np = cleaned.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (np.clip(cleaned_np, 0, 1) * 255.0).round().astype(np.uint8)
        except Exception:
            return image_np_uint8

    def prepare_inpaint_conditioning(self, positive, negative, latent_samples, resized_mask):
        conditioned_positive = []
        for p in positive:
            new_cond_dict = p[1].copy()
            new_cond_dict["concat_latent_image"] = latent_samples
            new_cond_dict["concat_mask"] = resized_mask
            conditioned_positive.append([p[0], new_cond_dict])
        
        conditioned_negative = []
        for n in negative:
            new_cond_dict = n[1].copy()
            new_cond_dict["concat_latent_image"] = latent_samples
            new_cond_dict["concat_mask"] = resized_mask
            conditioned_negative.append([n[0], new_cond_dict])
            
        return conditioned_positive, conditioned_negative

    def run_ksampler(self, model, positive, negative, latent, steps, cfg, sampler_name, scheduler, denoise, seed, denoise_mask=None):
        try:
            check_for_interruption()
            device = model_management.get_torch_device()
            latent_image = latent["samples"]

            if denoise_mask is not None:
                denoise_mask = denoise_mask.to(device)

            try:
                noise = comfy.sample.prepare_noise(latent_image, seed, device=device)
            except TypeError:
                noise = comfy.sample.prepare_noise(latent_image, seed)
                noise = noise.to(device)
            
            if denoise_mask is not None:
                cond_mask = denoise_mask
                if len(cond_mask.shape) == 3:
                    cond_mask = cond_mask.unsqueeze(1)
                
                positive_cond, negative_cond = self.prepare_inpaint_conditioning(
                    positive, negative, latent_image, cond_mask
                )
            else:
                positive_cond = self.prepare_conditioning_for_sampling(positive, device)
                negative_cond = self.prepare_conditioning_for_sampling(negative, device)

            try:
                previewer = latent_preview.get_previewer(device, model.model.latent_format)
            except:
                previewer = None

            pbar = comfy.utils.ProgressBar(steps)

            def preview_callback(step, x0, x, total_steps):
                preview_bytes = None
                if previewer:
                    try:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    except:
                        pass
                pbar.update_absolute(step + 1, total_steps, preview_bytes)

            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond, latent_image,
                denoise=denoise, disable_noise=False, start_step=0, last_step=steps,
                force_full_denoise=False, noise_mask=denoise_mask, callback=preview_callback,
                disable_pbar=True, seed=seed
            )
            
            return {"samples": samples}

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in KSampler: {e}")
            import traceback
            traceback.print_exc()
            latent_shape = latent.get("samples", torch.zeros((1, 4, 64, 64))).shape
            return {"samples": torch.zeros(latent_shape, device=model_management.get_torch_device())}
        
    def prepare_conditioning_for_sampling(self, conditioning, device):
        try:
            if not conditioning: return conditioning
            prepared_conditioning = []
            for cond_item in conditioning:
                if cond_item[0].device == device: cond_tensor = cond_item[0]
                else: cond_tensor = cond_item[0].to(device)
                cond_dict = {}
                for key, value in cond_item[1].items():
                    if torch.is_tensor(value) and value.device != device: cond_dict[key] = value.to(device)
                    else: cond_dict[key] = value
                prepared_conditioning.append([cond_tensor, cond_dict])
            return prepared_conditioning
        except Exception:
            return conditioning

    def apply_color_correction(self, generated_np_uint8, original_crop_np_uint8):
        try:
            generated_lab = cv2.cvtColor(generated_np_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
            original_lab = cv2.cvtColor(original_crop_np_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            original_mean = np.mean(original_lab, axis=(0, 1))
            original_std = np.std(original_lab, axis=(0, 1))
            
            generated_mean = np.mean(generated_lab, axis=(0, 1))
            generated_std = np.std(generated_lab, axis=(0, 1))
            
            corrected_lab = generated_lab.copy()
            for i in range(3):
                if generated_std[i] > 1e-6:
                    corrected_lab[:, :, i] = ((generated_lab[:, :, i] - generated_mean[i]) / generated_std[i]) * original_std[i] + original_mean[i]
                else:
                    corrected_lab[:, :, i] = generated_lab[:, :, i] - generated_mean[i] + original_mean[i]
            
            corrected_lab = np.clip(corrected_lab, 0, 255)
            corrected_rgb = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            return corrected_rgb
            
        except Exception as e:
            print(f"Error in color correction: {e}")
            return generated_np_uint8

    def create_compositing_mask(self, mask_np, blend_softness, crop_h, crop_w):
        mask_size = max(crop_h, crop_w)
        blend_ratio = blend_softness / 400.0
        
        core_mask = (mask_np > 0.5).astype(np.float32)
        
        expand_amount = int(mask_size * blend_ratio * 0.8)
        expanded_mask = core_mask
        if expand_amount > 0:
            kernel_size = expand_amount * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            expanded_mask = cv2.dilate(core_mask, kernel, iterations=1)

        blur_amount = int(mask_size * blend_ratio * 1.2)
        blur_kernel_size = max(3, blur_amount * 2 + 1)
        sigma = blur_kernel_size / 3.0
        
        blurred_mask = cv2.GaussianBlur(expanded_mask, (blur_kernel_size, blur_kernel_size), sigma)
        
        final_mask = np.maximum(core_mask, blurred_mask)

        fade_pixels = int(min(crop_h, crop_w) * 0.02) 
        fade_pixels = max(8, fade_pixels)
        
        if fade_pixels > 0:
            y_grad = np.linspace(0, 1, fade_pixels)
            x_grad = np.linspace(0, 1, fade_pixels)
            
            final_mask[:fade_pixels, :] *= y_grad[:, np.newaxis]
            final_mask[-fade_pixels:, :] *= y_grad[::-1][:, np.newaxis]
            final_mask[:, :fade_pixels] *= x_grad[np.newaxis, :]
            final_mask[:, -fade_pixels:] *= x_grad[::-1][np.newaxis, :]

        final_mask[:1, :] = 0
        final_mask[-1:, :] = 0
        final_mask[:, :1] = 0
        final_mask[:, -1:] = 0
            
        return np.clip(final_mask, 0, 1)
    
    def process_inpaint(self, model, vae, positive, negative, image, mask, steps, cfg, sampler, scheduler, 
                       denoise, seed, processing_resolution, manual_rotation, enable_pre_upscale, upscaler_model,
                       mask_expansion, sampling_mask_blur_size, sampling_mask_blur_strength, blend_softness,
                       enable_color_correction, enable_differential_diffusion, crop_padding, bypass_cropping):
        try:
            check_for_interruption()
            device = model_management.get_torch_device()
            
            if bypass_cropping:
                image_tensor = image.to(device)
                mask_tensor = mask.to(device)
                
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                elif len(mask_tensor.shape) == 3:
                    mask_tensor = mask_tensor.unsqueeze(1)
                
                VAEEncodeClass = nodes.NODE_CLASS_MAPPINGS['VAEEncode']
                vae_encoder = VAEEncodeClass()
                latent = vae_encoder.encode(vae, image_tensor)[0]
                
                latent_h, latent_w = latent["samples"].shape[2], latent["samples"].shape[3]
                sampling_mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
                
                if sampling_mask_blur_size > 1 and sampling_mask_blur_strength > 0:
                    ksize = sampling_mask_blur_size if sampling_mask_blur_size % 2 == 1 else sampling_mask_blur_size + 1
                    base_sigma = (ksize - 1) / 8.0
                    strength_tensor = torch.tensor(sampling_mask_blur_strength - 1.0, device=device, dtype=torch.float32)
                    multiplier = 1.0 + torch.tanh(strength_tensor) * 2.0
                    actual_sigma = base_sigma * multiplier.item()
                    
                    sampling_mask_latent = kornia.filters.gaussian_blur2d(
                        sampling_mask_latent, (ksize, ksize), (actual_sigma, actual_sigma)
                    )
                
                if enable_differential_diffusion:
                    model_clone = model.clone()
                    model_clone.set_model_denoise_mask_function(self.differential_diffusion_function)
                else:
                    model_clone = model
                
                sampled_latent = self.run_ksampler(
                    model_clone, positive, negative, latent,
                    steps, cfg, sampler, scheduler, denoise, seed,
                    denoise_mask=sampling_mask_latent
                )
                
                VAEDecodeClass = nodes.NODE_CLASS_MAPPINGS['VAEDecode']
                vae_decoder = VAEDecodeClass()
                result_tensor = vae_decoder.decode(vae, sampled_latent)[0]
                
                if result_tensor.min() < -0.5:
                    result_tensor = (result_tensor + 1.0) / 2.0
                result_tensor = torch.clamp(result_tensor, 0.0, 1.0)
                
                return (result_tensor, sampled_latent)
            
            cropped_face_tensor, sampler_mask_tensor, restore_info = self.mask_processor.process_and_crop(
                image_tensor=image,
                mask_tensor=mask,
                crop_padding=crop_padding,
                processing_resolution=processing_resolution,
                mask_expansion=mask_expansion,
                enable_pre_upscale=enable_pre_upscale,
                upscaler_model_name=upscaler_model,
                upscaler_loader_callback=self.load_upscaler_model,
                upscaler_run_callback=self.upscale_image_wrapper
            )
            
            if restore_info.get("original_image_size") == (0, 0):
                return (image, torch.zeros((1, 4, 64, 64), device=device))

            if manual_rotation != "None":
                image_np = (cropped_face_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                rotated_image = self.apply_manual_rotation(image_np, manual_rotation)
                cropped_face_tensor = torch.from_numpy(rotated_image.astype(np.float32) / 255.0).unsqueeze(0)

                mask_np = sampler_mask_tensor.squeeze().cpu().numpy()
                rotated_mask = self.apply_manual_rotation(mask_np, manual_rotation)
                sampler_mask_tensor = torch.from_numpy(rotated_mask).unsqueeze(0)

            cropped_face_tensor = cropped_face_tensor.to(device)
            sampler_mask_tensor = sampler_mask_tensor.to(device)
            
            if len(sampler_mask_tensor.shape) == 3:
                sampler_mask_tensor = sampler_mask_tensor.unsqueeze(1)
            
            VAEEncodeClass = nodes.NODE_CLASS_MAPPINGS['VAEEncode']
            vae_encoder = VAEEncodeClass()
            latent = vae_encoder.encode(vae, cropped_face_tensor)[0]
            
            latent_h_real, latent_w_real = latent["samples"].shape[2], latent["samples"].shape[3]
            
            sampling_mask_latent = F.interpolate(sampler_mask_tensor, size=(latent_h_real, latent_w_real), mode='bilinear', align_corners=False)
            
            if sampling_mask_blur_size > 1 and sampling_mask_blur_strength > 0:
                ksize = sampling_mask_blur_size if sampling_mask_blur_size % 2 == 1 else sampling_mask_blur_size + 1
                base_sigma = (ksize - 1) / 8.0
                strength_tensor = torch.tensor(sampling_mask_blur_strength - 1.0, device=device, dtype=torch.float32)
                multiplier = 1.0 + torch.tanh(strength_tensor) * 2.0
                actual_sigma = base_sigma * multiplier.item()

                sampling_mask_latent = kornia.filters.gaussian_blur2d(
                    sampling_mask_latent, (ksize, ksize), (actual_sigma, actual_sigma)
                )

            if enable_differential_diffusion:
                model_clone = model.clone()
                model_clone.set_model_denoise_mask_function(self.differential_diffusion_function)
            else:
                model_clone = model
            
            sampled_latent = self.run_ksampler(
                model_clone, positive, negative, latent,
                steps, cfg, sampler, scheduler, denoise, seed,
                denoise_mask=sampling_mask_latent
            )
            
            VAEDecodeClass = nodes.NODE_CLASS_MAPPINGS['VAEDecode']
            vae_decoder = VAEDecodeClass()
            generated_tensor = vae_decoder.decode(vae, sampled_latent)[0]
            
            if generated_tensor.min() < -0.5:
                generated_tensor = (generated_tensor + 1.0) / 2.0
            generated_tensor = torch.clamp(generated_tensor, 0.0, 1.0)
            
            generated_np = generated_tensor.cpu().numpy()[0]
            generated_np_uint8 = (np.clip(generated_np, 0, 1) * 255.0).round().astype(np.uint8)
            
            orig_crop_w, orig_crop_h = restore_info['original_crop_size']
            crop_x1, crop_y1, crop_x2, crop_y2 = restore_info['crop_coords']
            original_image_full = restore_info['original_image']
            
            if len(original_image_full.shape) == 4:
                original_image_np = original_image_full[0]
            else:
                original_image_np = original_image_full

            if enable_color_correction:
                orig_img_uint8 = (np.clip(original_image_np, 0, 1) * 255.0).astype(np.uint8)
                original_crop = orig_img_uint8[crop_y1:crop_y2, crop_x1:crop_x2]
                
                original_crop_resized = cv2.resize(original_crop, (generated_np_uint8.shape[1], generated_np_uint8.shape[0]), interpolation=cv2.INTER_AREA)
                generated_np_uint8 = self.apply_color_correction(generated_np_uint8, original_crop_resized)
            
            if manual_rotation != "None":
                generated_np_uint8 = self.reverse_manual_rotation(generated_np_uint8, manual_rotation)
            
            generated_final = cv2.resize(generated_np_uint8, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_AREA)
            
            blend_mask_orig = restore_info['blend_mask']
            blend_mask_crop = blend_mask_orig[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if blend_mask_crop.shape[:2] != (orig_crop_h, orig_crop_w):
                blend_mask_crop = cv2.resize(blend_mask_crop, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_LINEAR)

            final_blend_mask = self.create_compositing_mask(blend_mask_crop, blend_softness, orig_crop_h, orig_crop_w)
            
            mask_3ch = np.stack([final_blend_mask] * 3, axis=-1)
            
            result_image = (np.clip(original_image_np, 0, 1) * 255.0).astype(np.uint8).copy()
            current_area = result_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if current_area.shape[:2] != generated_final.shape[:2]:
                generated_final = cv2.resize(generated_final, (current_area.shape[1], current_area.shape[0]), interpolation=cv2.INTER_AREA)
                mask_3ch = cv2.resize(mask_3ch, (current_area.shape[1], current_area.shape[0]), interpolation=cv2.INTER_NEAREST)

            blended_area = (generated_final * mask_3ch + current_area * (1 - mask_3ch)).astype(np.uint8)
            result_image[crop_y1:crop_y2, crop_x1:crop_x2] = blended_area
            
            result_tensor = torch.from_numpy(result_image.astype(np.float32) / 255.0).unsqueeze(0)
            result_latent = vae_encoder.encode(vae, result_tensor)[0]
            
            return (result_tensor, result_latent)
            
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"Error in inpaint processing: {e}")
            import traceback
            traceback.print_exc()
            return (image, vae.encode(image))