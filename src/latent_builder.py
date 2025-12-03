import torch
import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

class LatentBuilder:

    def __init__(self):
        pass
    
    RESOLUTIONS = {
        "[SDXL] 1024x1024 (Square 1:1)": (1024, 1024),
        "[SDXL] 896x1152 (Portrait 4:5)": (896, 1152),
        "[SDXL] 832x1216 (Portrait 3:4)": (832, 1216),
        "[SDXL] 768x1344 (Portrait ~9:15)": (768, 1344),
        "[SDXL] 864x1536 (Portrait 9:16)": (864, 1536),
        "[SDXL] 768x1536 (Portrait 2:1)": (768, 1536),
        "[SDXL] 896x1264 (A4 Portrait)": (896, 1264),
        "[SDXL] 1264x896 (A4 Landscape)": (1264, 896),
        "[SDXL] 1152x896 (Landscape 5:4)": (1152, 896),
        "[SDXL] 1216x832 (Landscape 4:3)": (1216, 832),
        "[SDXL] 1344x768 (Landscape 16:9)": (1344, 768),
        "[SDXL] 1536x640 (Landscape 21:9)": (1536, 640),
        "[SDXL] 2048x768 (Ultra-Wide 32:9)": (2048, 768),
        "[SDXL] 1920x800 (Cinematic 2.40:1)": (1920, 800),

        "[SD1.5] 512x512 (Square 1:1)": (512, 512),
        "[SD1.5] 512x768 (Portrait 2:3)": (512, 768),
        "[SD1.5] 384x640 (Tall 9:16)": (384, 640),
        "[SD1.5] 640x384 (Widescreen 16:9)": (640, 384),
        "[SD1.5] 768x512 (Landscape 3:2)": (768, 512),
        "[SD1.5] 640x480 (Landscape 4:3)": (640, 480),
        "[SD1.5] 640x512 (Landscape 5:4)": (640, 512),

        "[SD2.1] 768x768 (Square 1:1)": (768, 768),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "self_correction": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Performs a final low-denoise polishing pass to fix small artifacts."}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                
                "resolution_preset": (["Custom"] + list(cls.RESOLUTIONS.keys()),),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("LATENT", "IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Forbidden Vision"

    def sample(self, model, positive, negative, self_correction, seed, steps, cfg, sampler_name, scheduler,
             resolution_preset, custom_width, custom_height, batch_size, vae=None):
        
        if resolution_preset == "Custom": 
            width, height = custom_width, custom_height
        else: 
            width, height = self.RESOLUTIONS[resolution_preset]

        width = (width // 8) * 8
        height = (height // 8) * 8

        device = model_management.get_torch_device()
        
        latent_tensor = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
        blank_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
        
        try:
            result_tensor = self._standard_sampling(
                model, positive, negative, latent_tensor, seed, steps, cfg, 
                sampler_name, scheduler, device
            )
            
            final_latent = {"samples": result_tensor}

            if self_correction:
                sampler_info = {
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "seed": seed + 1
                }
                final_latent = self._final_polish_pass(final_latent, model, positive, negative, sampler_info)
            
            if vae is not None:
                image_out = vae.decode(final_latent["samples"])
                return (final_latent, image_out,)
            else:
                return (final_latent, blank_image,)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"❌ Error during sampling: {e}")
            return ({"samples": latent_tensor}, blank_image,)
        
    def prepare_conditioning(self, conditioning, device):
        if not conditioning: return []
        prepared = []
        for cond_item in conditioning:
            model_management.throw_exception_if_processing_interrupted()
            cond_tensor = cond_item[0].to(device)
            cond_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_item[1].items()}
            prepared.append([cond_tensor, cond_dict])
        return prepared

    def _standard_sampling(self, model, positive_cond, negative_cond, latent_tensor, seed, steps, cfg, sampler_name, scheduler, device):
        positive = self.prepare_conditioning(positive_cond, device)
        negative = self.prepare_conditioning(negative_cond, device)
        noise = comfy.sample.prepare_noise(latent_tensor, seed)

        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)
        
        def callback(step, x0, x, total_steps):
            if previewer:
                preview_image = previewer.decode_latent_to_preview_image("JPEG", x0)
                pbar.update_absolute(step + 1, total_steps, preview_image)
            else:
                pbar.update_absolute(step + 1, total_steps, None)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_tensor, start_step=0, last_step=steps, force_full_denoise=True, callback=callback, disable_pbar=False)
        return samples

    def _final_polish_pass(self, latent_dict, model, positive, negative, sampler_info):
        POLISH_DENOISE = 0.05
        POLISH_STEPS = 2
        POLISH_CFG = 1.0
        
        device = model_management.get_torch_device()
        positive = self.prepare_conditioning(positive, device)
        negative = self.prepare_conditioning(negative, device)
        
        latent_to_polish = latent_dict["samples"]
        
        sampler = comfy.samplers.KSampler(
            model, 
            steps=POLISH_STEPS, 
            device=device, 
            sampler=sampler_info["sampler_name"], 
            scheduler=sampler_info["scheduler"], 
            denoise=POLISH_DENOISE, 
            model_options=model.model_options
        )
        
        noise = comfy.sample.prepare_noise(latent_to_polish, sampler_info["seed"])

        polished_latent = sampler.sample(
            noise, 
            positive, 
            negative, 
            cfg=POLISH_CFG, 
            latent_image=latent_to_polish, 
            start_step=0, 
            last_step=POLISH_STEPS, 
            force_full_denoise=True,
            disable_pbar=True
        )
        
        return {"samples": polished_latent}