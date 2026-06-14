import torch
import math
import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from .utils import RESOLUTIONS

class LatentBuilder:

    def __init__(self):
        pass
    
    RESOLUTIONS = RESOLUTIONS
    
    @classmethod
    def INPUT_TYPES(cls):
        schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        samplers = list(comfy.samplers.KSampler.SAMPLERS)

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "self_correction": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Performs a final low-denoise polishing pass to fix small artifacts."}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler_ancestral"}),
                "scheduler": (schedulers, {"default": "sgm_uniform"}),
                
                "resolution_preset": (["Custom"] + list(cls.RESOLUTIONS.keys()),),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "cfg_mode": (["Constant", "Linear", "Ease Down"], {"default": "Constant", "tooltip": "Constant: fixed CFG. Linear: straight transition to CFG Finish. Ease Down: fast move toward CFG Pivot, then gentle settle to CFG Finish."}),
                "cfg_finish": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1, "tooltip": "Target CFG value at the final step. Only used when CFG Mode is not Constant."}),
                "cfg_pivot": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "CFG value the curve moves toward quickly before settling to CFG Finish. Only used in Ease Down mode. When CFG > CFG Finish, the curve drops fast toward this value. When CFG < CFG Finish, it rises fast toward it."
                }),
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
            resolution_preset, custom_width, custom_height, batch_size, cfg_mode="Constant", cfg_finish=5.5, cfg_pivot=4.0, vae=None):
        
        if resolution_preset == "Custom": 
            width, height = custom_width, custom_height
        else: 
            width, height = self.RESOLUTIONS[resolution_preset]

        width = (width // 8) * 8
        height = (height // 8) * 8

        device = model_management.get_torch_device()

        from .utils import make_empty_latent
        latent_tensor = make_empty_latent(batch_size, height, width, model, device)
        blank_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
        
        try:
            result_tensor = self._standard_sampling(
                model, positive, negative, latent_tensor, seed, steps, cfg,
                sampler_name, scheduler, device, cfg_mode, cfg_finish, cfg_pivot
            )
            
            final_latent = {"samples": result_tensor}

            if self_correction:
                polish_cfg = cfg if cfg_mode == "Constant" else cfg_finish
                sampler_info = {
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "seed": seed + 1,
                    "polish_cfg": min(polish_cfg, 2.0)  # cap it low so the polish pass stays gentle
                }
                final_latent = self._final_polish_pass(final_latent, model, positive, negative, sampler_info)
            
            if vae is not None:
                from .utils import normalize_vae_decode_output
                image_out = normalize_vae_decode_output(vae.decode(final_latent["samples"]))
                return (final_latent, image_out,)
            else:
                return (final_latent, blank_image,)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"❌ Error during sampling: {e}")
            return ({"samples": latent_tensor}, blank_image,)
    
    def _get_cfg_at_step(self, step, total_steps, cfg_start, cfg_finish, cfg_mode, cfg_pivot=4.0):
        if cfg_mode == "Constant" or total_steps <= 1:
            return float(cfg_start)

        t = min(max(step / (total_steps - 1), 0.0), 1.0)
        cfg_start = float(cfg_start)
        cfg_finish = float(cfg_finish)

        if cfg_mode == "Linear":
            return cfg_start + (cfg_finish - cfg_start) * t

        # Ease Down mode
        cfg_pivot = max(0.1, float(cfg_pivot))

        if cfg_finish < cfg_start:
            # Dropping: fast toward pivot, then gentle settle to finish
            pivot = max(cfg_finish, cfg_pivot)
            to_pivot = 1.0 - ((1.0 - t) ** 3.0)
            to_finish = t ** 2.0
            current = cfg_start + (pivot - cfg_start) * to_pivot
            return current + (cfg_finish - pivot) * to_finish
        else:
            # Rising: fast toward pivot, then gentle settle to finish
            pivot = min(cfg_finish, cfg_pivot)
            to_pivot = 1.0 - ((1.0 - t) ** 3.0)
            to_finish = t ** 2.0
            current = cfg_start + (pivot - cfg_start) * to_pivot
            return current + (cfg_finish - pivot) * to_finish
    
    def prepare_conditioning(self, conditioning, device):
        if not conditioning: return []
        prepared = []
        for cond_item in conditioning:
            model_management.throw_exception_if_processing_interrupted()
            cond_tensor = cond_item[0].to(device)
            cond_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_item[1].items()}
            prepared.append([cond_tensor, cond_dict])
        return prepared

    def _standard_sampling(self, model, positive_cond, negative_cond, latent_tensor, seed, steps, cfg, sampler_name, scheduler, device, cfg_mode="Constant", cfg_finish=None, cfg_pivot=4.0):
        if cfg_finish is None:
            cfg_finish = cfg

        positive = self.prepare_conditioning(positive_cond, device)
        negative = self.prepare_conditioning(negative_cond, device)
        noise = comfy.sample.prepare_noise(latent_tensor, seed)

        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)

        # Always work on a copied dict so we don't mutate shared model state
        original_options = dict(model.model_options) if model.model_options else {}
        patched_options = dict(original_options)

        # Track the REAL sampler step from the callback
        step_state = {"current_step": 0}

        def callback(step, x0, x, total_steps):
            step_state["current_step"] = step
            if previewer:
                preview_image = previewer.decode_latent_to_preview_image("JPEG", x0)
                pbar.update_absolute(step + 1, total_steps, preview_image)
            else:
                pbar.update_absolute(step + 1, total_steps, None)

        if cfg_mode != "Constant":
            # Critical fix:
            # If the base cfg passed into sampler.sample() is exactly 1.0,
            # Comfy can skip unconditional guidance unless this is forced on.
            patched_options["disable_cfg1_optimization"] = True

            def dynamic_cfg(args):
                cond = args["cond"]
                uncond = args["uncond"]

                # Use the actual sampler step from callback, not function-call count
                current_step = min(step_state["current_step"], steps - 1)
                step_cfg = self._get_cfg_at_step(
                    current_step,
                    steps,
                    cfg,
                    cfg_finish,
                    cfg_mode,
                    cfg_pivot
                )

                # Safety clamp to avoid weird negative / zero behavior
                step_cfg = max(0.0, float(step_cfg))

                # Comfy expects sampler_cfg_function to return the "guidance-space" tensor,
                # not the final denoised prediction.
                return uncond + step_cfg * (cond - uncond)

            patched_options["sampler_cfg_function"] = dynamic_cfg

        # Temporarily patch model options
        model.model_options = patched_options

        try:
            sampler = comfy.samplers.KSampler(
                model,
                steps=steps,
                device=device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=1.0,
                model_options=model.model_options
            )

            samples = sampler.sample(
                noise,
                positive,
                negative,
                cfg=cfg,  # keep this as the starting/base cfg
                latent_image=latent_tensor,
                start_step=0,
                last_step=steps,
                force_full_denoise=True,
                callback=callback,
                disable_pbar=False
            )
        finally:
            # Always restore original options
            model.model_options = original_options

        return samples

    def _final_polish_pass(self, latent_dict, model, positive, negative, sampler_info):
        POLISH_DENOISE = 0.05
        POLISH_STEPS = 2
        POLISH_CFG = sampler_info.get("polish_cfg", 1.0)
        
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