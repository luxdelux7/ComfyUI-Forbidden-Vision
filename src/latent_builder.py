import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import math

class LatentBuilder:

    def __init__(self):
        self.run_summary_data = {}
    
    RESOLUTIONS = {
        "1024x1024 (Square 1:1)": (1024, 1024),
        "896x1152 (Portrait 4:5)": (896, 1152),
        "832x1216 (Portrait 3:4)": (832, 1216),
        "768x1344 (Portrait 9:16)": (768, 1344),
        "1152x896 (Landscape 5:4)": (1152, 896),
        "1216x832 (Landscape 4:3)": (1216, 832),
        "1344x768 (Landscape 16:9)": (1344, 768),
        "1536x640 (Landscape 21:9)": (1536, 640),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "self_correction": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                
                # --- CFG DECAY INPUTS ---
                "cfg_decay_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cfg_decay_start_at": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cfg_decay_curve": (["Smooth", "Linear", "Slow Start", "Fast Start"], {"default": "Smooth"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.1}),
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

    def sample(self, model, positive, negative, self_correction, cfg_decay_strength, cfg_decay_start_at, cfg_decay_curve,
            seed, steps, cfg, sampler_name, scheduler, resolution_preset, custom_width, custom_height, batch_size, vae=None):
        
        if resolution_preset == "Custom": 
            width, height = custom_width, custom_height
        else: 
            width, height = self.RESOLUTIONS[resolution_preset]

        device = model_management.get_torch_device()
        latent_tensor = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
        
        # Create a blank, black image tensor to use as a fallback output.
        # This prevents downstream nodes from crashing if no VAE is connected.
        blank_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
        
        final_latent = {"samples": latent_tensor}

        try:
            if cfg_decay_strength > 0:
                print(f"🧠 Generating at {width}x{height} with CFG Decay Schedule. Strength: {cfg_decay_strength:.2f}, Curve: {cfg_decay_curve}")
                result_tensor = self.adaptive_cfg_sampling(model, positive, negative, latent_tensor, seed, steps, cfg, sampler_name, scheduler, device, cfg_decay_strength, cfg_decay_start_at, cfg_decay_curve)
            else:
                print(f"🎨 Generating at {width}x{height} with Standard Static CFG: {cfg:.2f}")
                result_tensor = self.standard_sampling(model, positive, negative, latent_tensor, seed, steps, cfg, sampler_name, scheduler, device)
            
            initial_latent = {"samples": result_tensor}
            final_latent = initial_latent

            if self_correction:
                print("✨ Applying a final, low-denoise polish pass...")
                sampler_info = {
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "seed": seed + 1
                }
                polished_latent = self._final_polish_pass(initial_latent, model, positive, negative, sampler_info)
                print("✅ Polish complete.")
                final_latent = polished_latent
            
            if vae is not None:
                print("Decoding latent to image...")
                image_out = vae.decode(final_latent["samples"])
                print("✅ Decode complete.")
                return (final_latent, image_out,)
            else:
                # If no VAE is provided, return the generated latent and the blank image.
                return (final_latent, blank_image,)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            print(f"❌ Error during sampling: {e}")
            import traceback
            traceback.print_exc()
            # On any error, return the initial empty latent and the blank image.
            return ({"samples": latent_tensor}, blank_image,)
        
    def prepare_conditioning(self, conditioning, device):
        if not conditioning: return []
        prepared = []
        for cond_item in conditioning:
            model_management.throw_exception_if_processing_interrupted() # Check for interruption
            cond_tensor = cond_item[0].to(device)
            cond_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_item[1].items()}
            prepared.append([cond_tensor, cond_dict])
        return prepared

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

    def standard_sampling(self, model, positive_cond, negative_cond, latent_tensor, seed, steps, cfg, sampler_name, scheduler, device):
        positive = self.prepare_conditioning(positive_cond, device)
        negative = self.prepare_conditioning(negative_cond, device)
        noise = comfy.sample.prepare_noise(latent_tensor, seed)

        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            preview_image = previewer.decode_latent_to_preview_image("JPEG", x0)
            pbar.update_absolute(step + 1, total_steps, preview_image)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_tensor, start_step=0, last_step=steps, force_full_denoise=True, callback=callback, disable_pbar=False)
        return samples
    
    def adaptive_cfg_sampling(self, model, positive_cond, negative_cond, latent_tensor, seed, steps, base_cfg, sampler_name, scheduler, device, decay_strength, decay_start_at, decay_curve):
        positive = self.prepare_conditioning(positive_cond, device)
        negative = self.prepare_conditioning(negative_cond, device)
        noise = comfy.sample.prepare_noise(latent_tensor, seed)
        
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        
        # Pass the decay_curve parameter to the function that creates the scheduled model
        scheduled_model = self.create_adaptive_scheduled_model(model, base_cfg, steps, sigmas, decay_strength, decay_start_at, decay_curve)
        sampler.model = scheduled_model
        
        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            preview_image = previewer.decode_latent_to_preview_image("JPEG", x0)
            pbar.update_absolute(step + 1, total_steps, preview_image)

        samples = sampler.sample(noise, positive, negative, cfg=base_cfg, latent_image=latent_tensor, start_step=0, last_step=steps, force_full_denoise=True, callback=callback, disable_pbar=False, sigmas=sigmas)
        return samples

    def create_adaptive_scheduled_model(self, model, base_cfg, total_steps, sigmas, decay_strength, decay_start_at, decay_curve):
        current_step = [0]
        self.run_summary_data = {}

        def adaptive_cfg_function(args):
            model_management.throw_exception_if_processing_interrupted()
            
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            
            # Pass the decay_curve parameter to the calculation function
            final_cfg = self.calculate_adaptive_cfg(current_step[0], total_steps, base_cfg, self.run_summary_data, decay_strength, decay_start_at, decay_curve)
            
            log_parts = [f"Step {current_step[0] + 1}/{total_steps}: CFG {base_cfg:.2f} → {final_cfg:.2f}"]
            final_output = uncond_pred + final_cfg * (cond_pred - uncond_pred)
            
            post_stats = { "cfg": final_cfg }
            log_parts.append(f"[μ={torch.mean(torch.abs(final_output)).item():.2f}, σ={torch.std(final_output).item():.2f}]")
            
            self.run_summary_data.setdefault("step_history", []).append(post_stats)
            print(" | ".join(log_parts))
            
            current_step[0] += 1
            return args["input"] - final_output
        
        patched_model = model.clone()
        patched_model.set_model_sampler_cfg_function(adaptive_cfg_function)
        return patched_model
    
    def calculate_adaptive_cfg(self, step, total_steps, base_cfg, schedule_params, decay_strength, decay_start_at, decay_curve):
        start_decay_step = int(total_steps * decay_start_at)

        if step < start_decay_step:
            return base_cfg

        if "final_target_cfg" not in schedule_params:
            min_target_cfg = max(2.0, base_cfg / 2.5)
            end_cfg = base_cfg + (min_target_cfg - base_cfg) * decay_strength
            schedule_params["final_target_cfg"] = end_cfg
            print(f"🚀 CFG Decay initiated at step {step + 1}/{total_steps}. Base: {base_cfg:.2f}, Target: {end_cfg:.2f} (Strength: {decay_strength:.2f})")

        end_cfg = schedule_params["final_target_cfg"]
        
        total_decay_steps = total_steps - start_decay_step
        steps_into_decay = step - start_decay_step
        
        progress = 0.0
        if total_decay_steps > 1:
            progress = steps_into_decay / (total_decay_steps - 1)
        elif steps_into_decay > 0:
            progress = 1.0

        # --- CURVE LOGIC WITH NEW NAMES ---
        eased_progress = progress
        if decay_curve == "Smooth":
            # A gentle start and a gentle end (Sinusoidal)
            eased_progress = (math.cos(math.pi * progress) - 1) / -2.0
        elif decay_curve == "Slow Start":
            # Starts slow, accelerates towards the end (Quadratic In)
            eased_progress = progress * progress
        elif decay_curve == "Fast Start":
            # Starts fast, decelerates towards the end (Quadratic Out)
            eased_progress = 1 - (1 - progress) * (1 - progress)
        # "Linear" requires no change

        # Linearly interpolate from the base_cfg to the end_cfg using the *eased* progress.
        current_cfg = base_cfg + (end_cfg - base_cfg) * eased_progress
        
        return max(1.0, current_cfg)