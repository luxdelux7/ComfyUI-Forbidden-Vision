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
                "restart_mode": ([
                    "disabled",
                    "maximum detail",
                    "balanced detail",
                    "subtle detail",
                ], {"default": "disabled", "tooltip": "Restart position affects detail strength. Maximum=most aggressive changes, Balanced=good tradeoff, Subtle=gentle refinement. All add ~2-3 steps. Best with denoise 0.2-0.4."}),
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
                sampler_name, scheduler, restart_mode="disabled", vae=None):
        
        device = model_management.get_torch_device()
        input_latent = latent["samples"]
        blank_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
        
        restart_enabled = restart_mode != "disabled"
        
        try:
            positive_prep = self.prepare_conditioning(positive, device)
            negative_prep = self.prepare_conditioning(negative, device)
            
            noise = comfy.sample.prepare_noise(input_latent, seed)
            
            if restart_enabled:
                samples = self._rebuild_with_restart(
                    model, input_latent, noise, positive_prep, negative_prep,
                    seed, steps, cfg, denoise, sampler_name, scheduler, restart_mode, device
                )
            else:
                samples = self._rebuild_standard(
                    model, input_latent, noise, positive_prep, negative_prep,
                    seed, steps, cfg, denoise, sampler_name, scheduler, device
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
    
    def _rebuild_with_restart(self, model, input_latent, noise, positive, negative,
                              seed, steps, cfg, denoise, sampler_name, scheduler, restart_mode, device):

        model_sampling = model.get_model_object("model_sampling")
        total_steps = int(steps / denoise)
        full_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
        
        sigmas = full_sigmas[-(steps + 1):].to(device)
        
        restart_sigmas = self._get_lowdenoise_restart_schedule(sigmas, steps, denoise, restart_mode)
        
        low_level_sampler = comfy.samplers.sampler_object(sampler_name)
        
        total_pbar_steps = len(restart_sigmas) - 1
        x0_output = {}
        callback = latent_preview.prepare_callback(model, total_pbar_steps, x0_output)

        def restart_sampler_impl(model, x, sigmas, *args, **kwargs):
            chunks = self._split_sigmas_into_chunks(sigmas)
            
            outer_callback = kwargs.get("callback")
            current_step_offset = [0]

            def adjusted_callback(data):
                if 'i' in data:
                    data['i'] = current_step_offset[0] + data['i']
                if outer_callback:
                    outer_callback(data)
            
            kwargs["callback"] = adjusted_callback

            for i, chunk_sigmas in enumerate(chunks):
                chunk_sigmas = chunk_sigmas.to(x.device)
                
                if i > 0:
                    s_min = chunks[i-1][-1].item()
                    s_max = chunk_sigmas[0].item()
                    
                    if s_max > s_min:
                        noise_scale = (s_max**2 - s_min**2) ** 0.5
                        chunk_seed = seed + i * 100
                        noise_gen = torch.Generator(device=x.device).manual_seed(chunk_seed)
                        added_noise = torch.randn(x.shape, generator=noise_gen, device=x.device, dtype=x.dtype)
                        x = x + added_noise * noise_scale

                x = low_level_sampler.sampler_function(model, x, chunk_sigmas, *args, **kwargs)
                current_step_offset[0] += (len(chunk_sigmas) - 1)
                
            return x

        sampler_obj = comfy.samplers.KSAMPLER(
            restart_sampler_impl, 
            extra_options=low_level_sampler.extra_options,
            inpaint_options=low_level_sampler.inpaint_options
        )

        samples = comfy.sample.sample_custom(
            model, 
            noise, 
            cfg, 
            sampler_obj, 
            restart_sigmas, 
            positive, 
            negative, 
            input_latent, 
            noise_mask=None, 
            callback=callback,
            disable_pbar=False,
            seed=seed
        )

        return samples
    
    def _get_lowdenoise_restart_schedule(self, sigmas, steps, denoise, restart_mode):
        """
        Generate restart schedule based on restart position.
        
        Early restart (higher noise) = more aggressive detail changes
        Late restart (lower noise) = subtle polish/refinement
        
        Presets:
        - maximum detail: Early restart (25-50%), most aggressive
        - balanced detail: Mid restart (40-65%), good tradeoff
        - subtle detail: Late restart (55-80%), gentle refinement
        - extra quality: Broader restart (35-70%) with 3 steps
        """
        sigmas = sigmas.cpu()
        
        if "maximum" in restart_mode:
            n_steps = 2
            start_pct, end_pct = 0.25, 0.50
        elif "balanced" in restart_mode:
            n_steps = 2
            start_pct, end_pct = 0.40, 0.65
        elif "subtle" in restart_mode:
            n_steps = 2
            start_pct, end_pct = 0.55, 0.80
        elif "extra" in restart_mode:
            n_steps = 3
            start_pct, end_pct = 0.35, 0.70
        else:
            n_steps = 2
            start_pct, end_pct = 0.40, 0.65
        
        k_repeats = 1
        
        full_sigmas = []
        
        idx_start = int(len(sigmas) * start_pct)
        idx_end = int(len(sigmas) * end_pct)
        
        idx_start = max(0, min(idx_start, len(sigmas) - 2))
        idx_end = max(idx_start + 1, min(idx_end, len(sigmas) - 1))
        
        full_sigmas.extend(sigmas[0:idx_end + 1].tolist())
        
        s_max = sigmas[idx_start].item()
        s_min = sigmas[idx_end].item()
        
        restart_slice = torch.linspace(s_max, s_min, n_steps + 1)
        
        for _ in range(k_repeats):
            full_sigmas.extend(restart_slice.tolist())
        
        if idx_end + 1 < len(sigmas):
            full_sigmas.extend(sigmas[idx_end + 1:].tolist())
        
        return torch.tensor(full_sigmas)
    
    def _split_sigmas_into_chunks(self, sigmas):
        """Split sigmas into monotonic descending chunks at restart points"""
        chunks = []
        current_chunk = [sigmas[0]]
        for i in range(1, len(sigmas)):
            if sigmas[i] > sigmas[i-1]:
                chunks.append(torch.tensor(current_chunk))
                current_chunk = [sigmas[i]]
            else:
                current_chunk.append(sigmas[i])
        chunks.append(torch.tensor(current_chunk))
        return chunks
    
    def prepare_conditioning(self, conditioning, device):
        if not conditioning: return []
        prepared = []
        for cond_item in conditioning:
            model_management.throw_exception_if_processing_interrupted()
            cond_tensor = cond_item[0].to(device)
            cond_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_item[1].items()}
            prepared.append([cond_tensor, cond_dict])
        return prepared