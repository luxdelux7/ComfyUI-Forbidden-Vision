import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .src.utils import ensure_model_directories
from .src.face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .src.latent_refiner import LatentRefiner
from .src.latent_builder import LatentBuilder
ensure_model_directories()
NODE_CLASS_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": ForbiddenVisionFaceProcessorIntegrated,
    "LatentRefiner": LatentRefiner,
    "LatentBuilder": LatentBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": "Forbidden Vision - Fixer",
    "LatentRefiner": "Forbidden Vision - Refiner",
    "LatentBuilder": "Forbidden Vision - Builder",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# --- METADATA INJECTION PATCH ---
# This patch modifies the standard CLIPTextEncode node to include the original prompt
# text in its output. This allows downstream nodes like the Face Processor to
# access the prompt without requiring a dedicated conditioning node.

# 1. Import the original node class
from nodes import CLIPTextEncode

# 2. Store a reference to the original, untouched 'encode' method
original_clip_text_encode = CLIPTextEncode.encode

# 3. Define our new wrapper function
def fv_encode_wrapper(self, clip, text):
    # 4. First, call the original function to get the standard output.
    #    This ensures our patch is compatible with other custom nodes.
    encoded_output = original_clip_text_encode(self, clip, text)
    
    # 5. Now, add our non-destructive metadata to the conditioning's dictionary.
    #    We check to make sure the structure is what we expect.
    if isinstance(encoded_output, tuple) and len(encoded_output) > 0:
        conditioning = encoded_output[0]
        if isinstance(conditioning, list) and len(conditioning) > 0:
            if isinstance(conditioning[0], list) and len(conditioning[0]) == 2:
                if isinstance(conditioning[0][1], dict):
                    # Safely add our namespaced metadata
                    conditioning[0][1]["forbidden_vision_metadata"] = {"original_text": text}
                    
    # 6. Return the modified result
    return encoded_output

# 7. Finally, apply the patch by replacing the original method with our wrapper
CLIPTextEncode.encode = fv_encode_wrapper

print("[Forbidden Vision] Patched CLIPTextEncode to include prompt metadata for the Face Processor node.")