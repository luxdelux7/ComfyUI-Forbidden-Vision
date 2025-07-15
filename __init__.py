import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .src.utils import ensure_model_directories
from .src.face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .src.latent_ai_upscaler import LatentAIUpscaler

NODE_CLASS_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": ForbiddenVisionFaceProcessorIntegrated,
    "LatentAIUpscaler": LatentAIUpscaler, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": "Forbidden Vision - Face Processor",
    "LatentAIUpscaler": "Forbidden Vision - Latent AI Upscaler", 
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']