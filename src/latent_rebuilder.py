import torch
import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

class ForbiddenVisionRebuilder:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.28, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("LATENT", "IMAGE",)
    FUNCTION = "rebuild"
    CATEGORY = "Forbidden Vision"

    def rebuild(self, latent, model, positive, negative, seed, steps, cfg, denoise, 
                sampler_name, scheduler, vae=None):
        
        device = model_management.get_torch_device()
        input_latent = latent["samples"]
        blank_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
        
        try:
            positive_prep = self.prepare_conditioning(positive, device)
            negative_prep = self.prepare_conditioning(negative, device)
            
            noise = comfy.sample.prepare_noise(input_latent, seed)
            
            previewer = latent_preview.get_previewer(device, model.model.latent_format)
            pbar = comfy.utils.ProgressBar(steps)
            
            def callback(step, x0, x, total_steps):
                if previewer:
                    preview_image = previewer.decode_latent_to_preview_image("JPEG", x0)
                    pbar.update_absolute(step + 1, total_steps, preview_image)
                else:
                    pbar.update_absolute(step + 1, total_steps, None)
            
            sampler = comfy.samplers.KSampler(
                model, 
                steps=steps, 
                device=device, 
                sampler=sampler_name, 
                scheduler=scheduler, 
                denoise=denoise, 
                model_options=model.model_options
            )
            
            samples = sampler.sample(
                noise, 
                positive_prep, 
                negative_prep, 
                cfg=cfg, 
                latent_image=input_latent, 
                start_step=0, 
                last_step=steps, 
                force_full_denoise=True,
                callback=callback,
                disable_pbar=False
            )
            
            final_latent = {"samples": samples}
            
            if vae is not None:
                image_out = vae.decode(samples)
                return (final_latent, image_out,)
            else:
                return (final_latent, blank_image,)
                
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"❌ Error during rebuild: {e}")
            return ({"samples": input_latent}, blank_image,)
    
    def prepare_conditioning(self, conditioning, device):
        if not conditioning: return []
        prepared = []
        for cond_item in conditioning:
            model_management.throw_exception_if_processing_interrupted()
            cond_tensor = cond_item[0].to(device)
            cond_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_item[1].items()}
            prepared.append([cond_tensor, cond_dict])
        return prepared