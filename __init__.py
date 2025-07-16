import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .src.utils import ensure_model_directories
from .src.face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .src.latent_refiner import LatentRefiner
ensure_model_directories()
NODE_CLASS_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": ForbiddenVisionFaceProcessorIntegrated,
    "LatentRefiner": LatentRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": "Forbidden Vision - Face Processor",
    "LatentRefiner": "Forbidden Vision - Latent Refiner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']