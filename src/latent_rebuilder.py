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
        schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
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
                "sampler_name": (samplers, {"default": "euler_ancestral"}),
                "scheduler": (schedulers, {"default": "sgm_uniform"}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("LATENT", "IMAGE",)
    FUNCTION = "rebuild"
    CATEGORY = "Forbidden Vision"

    def prepare_conditioning(self, conditioning, device):
        """Move conditioning tensors to the target device."""
        prepped = []
        for cond, meta in conditioning:
            cond_out = cond.to(device) if isinstance(cond, torch.Tensor) else cond
            meta_out = {}
            for k, v in meta.items():
                meta_out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            prepped.append([cond_out, meta_out])
        return prepped

    def rebuild(self, latent, model, positive, negative, seed, steps, cfg, denoise,
                sampler_name, scheduler, vae=None):
        
        device = model_management.get_torch_device()
        input_latent = latent["samples"]

        # Reconcile latent layout with what the model expects
        from .utils import detect_model_latent_layout
        _, model_is_temporal = detect_model_latent_layout(model)
        if model_is_temporal and input_latent.dim() == 4:
            input_latent = input_latent.unsqueeze(2)
        elif not model_is_temporal and input_latent.dim() == 5 and input_latent.shape[2] == 1:
            input_latent = input_latent.squeeze(2)
        blank_image = torch.zeros((input_latent.shape[0], 1, 1, 3), dtype=torch.float32)
        
        try:
            positive_prep = self.prepare_conditioning(positive, device)
            negative_prep = self.prepare_conditioning(negative, device)
            
            noise = comfy.sample.prepare_noise(input_latent, seed)
            
            samples = self._rebuild_standard(
                model, input_latent, noise, positive_prep, negative_prep,
                seed, steps, cfg, denoise, sampler_name, scheduler, device
            )
            
            final_latent = {"samples": samples}
            
            if vae is not None:
                from .utils import normalize_vae_decode_output
                image_out = normalize_vae_decode_output(vae.decode(samples))
                return (final_latent, image_out,)
            else:
                return (final_latent, blank_image,)
                
        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"❌ Error during rebuild: {e}")
            import traceback
            traceback.print_exc()
            return ({"samples": input_latent}, blank_image,)
    
    def _rebuild_standard(self, model, input_latent, noise, positive, negative,
                          seed, steps, cfg, denoise, sampler_name, scheduler, device):
        """Standard rebuild using KSampler"""
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
            positive,
            negative,
            cfg=cfg,
            latent_image=input_latent,
            start_step=0,
            last_step=steps,
            force_full_denoise=True,
            callback=callback,
            disable_pbar=False
        )
        
        return samples