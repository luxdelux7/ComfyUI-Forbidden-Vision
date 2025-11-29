import comfy.samplers

class SamplerSchedulerSettings:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("sampler_name", "scheduler",)
    FUNCTION = "get_settings"
    CATEGORY = "Forbidden Vision"

    def get_settings(self, sampler_name, scheduler):
        return (sampler_name, scheduler,)