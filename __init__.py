from .utils import ensure_model_directories

ensure_model_directories()

from .face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .latent_ai_upscaler import LatentAIUpscaler

NODE_CLASS_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": ForbiddenVisionFaceProcessorIntegrated,
    "LatentAIUpscaler": LatentAIUpscaler, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": "Forbidden Vision - Face Processor",
    "LatentAIUpscaler": "Forbidden Vision - Latent AI Upscaler", 
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']