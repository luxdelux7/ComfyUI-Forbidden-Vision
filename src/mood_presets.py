PRESETS = {
    "Disabled": {},

    # --- Bright Natural ---

    "Summer Day": {
        "mood_type": "bright_natural",
        "color_palette": {
            "highlight_tint": [1.0, 0.99, 0.98],
            "shadow_tint": [0.75, 0.82, 0.95],
            "accent_colors": [[0.2, 0.6, 1.0], [0.3, 0.8, 0.4]]
        },
        "temperature_shift": "none",
        "brightness_shift": "enhance_existing",
        "atmospheric_effects": {
            "type": "clean_natural",
            "glow_colors": [[0.95, 0.98, 1.0]]
        }
    },

    "Lush Greenhouse": {
        "mood_type": "bright_natural",
        "color_palette": {
            "highlight_tint": [0.95, 1.0, 0.95],
            "shadow_tint": [0.3, 0.45, 0.35],
            "accent_colors": [[0.5, 0.8, 0.4], [0.6, 0.4, 0.3]]
        },
        "temperature_shift": "none",
        "brightness_shift": "enhance_existing",
        "atmospheric_effects": {
            "type": "clean_natural",
            "glow_colors": [[0.85, 1.0, 0.8]]
        }
    },

    "Crisp Autumn": {
        "mood_type": "bright_natural",
        "color_palette": {
            "highlight_tint": [1.0, 0.97, 0.92],
            "shadow_tint": [0.8, 0.85, 0.95],
            "accent_colors": [[1.0, 0.8, 0.6], [0.6, 0.7, 0.9], [0.9, 0.5, 0.3]]
        },
        "temperature_shift": "slightly_warmer",
        "brightness_shift": "enhance_existing",
        "atmospheric_effects": {
            "type": "clean_natural",
            "glow_colors": [[1.0, 0.96, 0.9]]
        }
    },

    "Overcast Blues": {
        "mood_type": "bright_natural",
        "color_palette": {
            "highlight_tint": [0.95, 0.98, 1.0],
            "shadow_tint": [0.6, 0.7, 0.8],
            "accent_colors": [[0.4, 0.7, 0.5], [0.8, 0.9, 0.9]]
        },
        "temperature_shift": "slightly_cooler",
        "brightness_shift": "enhance_existing",
        "atmospheric_effects": {
            "type": "clean_natural",
            "glow_colors": [[0.85, 0.9, 1.0]]
        }
    },

    # --- Warm Cinematic ---


    "Golden Hour": {
        "mood_type": "warm_cinematic",
        "color_palette": {
            "highlight_tint": [1.0, 0.98, 0.96],
            "shadow_tint": [0.3, 0.28, 0.27],
            "accent_colors": [[0.75, 0.55, 0.4], [0.65, 0.75, 0.5]]
        },
        "temperature_shift": "warmer",
        "brightness_shift": "darker_but_vibrant",
        "atmospheric_effects": {
            "type": "warm_cinematic",
            "glow_colors": [[1.0, 0.98, 0.96]]
        }
    },

    "Cozy Fireside": {
        "mood_type": "warm_cinematic",
        "color_palette": {
            "highlight_tint": [1.0, 0.85, 0.6],
            "shadow_tint": [0.2, 0.1, 0.05],
            "accent_colors": [[1.0, 0.7, 0.4]]
        },
        "temperature_shift": "much_warmer",
        "brightness_shift": "much_darker",
        "atmospheric_effects": {
            "type": "warm_cinematic",
            "glow_colors": [[1.0, 0.85, 0.55], [1.0, 0.6, 0.3]]
        }
    },


    # --- Cool Dramatic ---

    "Moonlit Night": {
        "mood_type": "cool_dramatic",
        "color_palette": {
            "highlight_tint": [0.95, 0.98, 1.05],
            "shadow_tint": [0.1, 0.15, 0.25],
            "accent_colors": [[0.6, 0.8, 1.0], [0.4, 0.6, 0.9], [0.8, 0.9, 1.0]]
        },
        "temperature_shift": "much_cooler",
        "brightness_shift": "much_darker",
        "atmospheric_effects": {
            "type": "mysterious_cool",
            "glow_colors": [[0.9, 0.95, 1.05], [0.4, 0.6, 0.9]]
        }
    },

    "Urban Twilight": {
        "mood_type": "cool_dramatic",
        "color_palette": {
            "highlight_tint": [1.0, 0.9, 0.85],
            "shadow_tint": [0.2, 0.25, 0.4],
            "accent_colors": [[0.7, 0.6, 0.9], [1.0, 0.8, 0.7]]
        },
        "temperature_shift": "cooler",
        "brightness_shift": "moodier",
        "atmospheric_effects": {
            "type": "mysterious_cool",
            "glow_colors": [[0.7, 0.75, 1.0], [1.0, 0.85, 0.75]]
        }
    },


    "Overcast Rain": {
        "mood_type": "cool_dramatic",
        "color_palette": {
            "highlight_tint": [0.85, 0.9, 1.0],
            "shadow_tint": [0.3, 0.35, 0.45],
            "accent_colors": [[0.5, 0.6, 0.7], [0.6, 0.7, 0.8]]
        },
        "temperature_shift": "cooler",
        "brightness_shift": "slightly_moodier",
        "atmospheric_effects": {
            "type": "mysterious_cool",
            "glow_colors": [[0.7, 0.8, 0.9]]
        }
    },

    # --- Cyberpunk Vibrant ---

    "Neon City": {
        "mood_type": "cyberpunk_vibrant",
        "color_palette": {
            "highlight_tint": [0.9, 0.7, 1.0],
            "shadow_tint": [0.15, 0.1, 0.2],
            "accent_colors": [[1.0, 0.2, 0.8], [0.2, 0.9, 1.0], [0.8, 0.3, 1.0], [1.0, 0.4, 0.6]]
        },
        "temperature_shift": "slightly_cooler",
        "brightness_shift": "darker_but_vibrant",
        "atmospheric_effects": {
            "type": "cyberpunk_complex",
            "glow_colors": [[1.0, 0.2, 0.8], [0.2, 0.9, 1.0], [0.8, 0.3, 1.0]],
            "complex_glows": True,
            "intensity_multiplier": 1.8,
            "saturation_boost": 1.8,
            "coverage_multiplier": 1.5
        }
    },

    "Retro Arcade": {
        "mood_type": "cyberpunk_vibrant",
        "color_palette": {
            "highlight_tint": [0.9, 1.0, 1.0],
            "shadow_tint": [0.1, 0.1, 0.15],
            "accent_colors": [[1.0, 0.1, 0.1], [0.1, 0.4, 1.0], [0.6, 1.0, 0.2]]
        },
        "temperature_shift": "none",
        "brightness_shift": "darker_but_vibrant",
        "atmospheric_effects": {
            "type": "cyberpunk_complex",
            "glow_colors": [[1.0, 0.2, 0.2], [0.2, 0.6, 1.0]],
            "complex_glows": True,
            "intensity_multiplier": 1.8,
            "saturation_boost": 1.6,
            "coverage_multiplier": 1.1
        }
    },

    "Holographic Glitch": {
        "mood_type": "cyberpunk_vibrant",
        "color_palette": {
            "highlight_tint": [0.9, 1.0, 0.95],
            "shadow_tint": [0.1, 0.05, 0.15],
            "accent_colors": [[0.5, 1.0, 0.8], [1.0, 0.6, 1.0]]
        },
        "temperature_shift": "cooler",
        "brightness_shift": "darker_but_vibrant",
        "atmospheric_effects": {
            "type": "cyberpunk_complex",
            "glow_colors": [[0.5, 1.0, 0.8], [1.0, 0.6, 1.0]],
            "complex_glows": True,
            "intensity_multiplier": 1.8,
            "saturation_boost": 1.5,
            "coverage_multiplier": 1.2
        }
    },

    "Night Parade": {
        "mood_type": "cyberpunk_vibrant",
        "color_palette": {
            "highlight_tint": [1.0, 0.9, 0.7],
            "shadow_tint": [0.1, 0.05, 0.1],
            "accent_colors": [[1.0, 0.5, 0.1], [1.0, 0.2, 0.2], [0.9, 0.6, 0.0]]
        },
        "temperature_shift": "slightly_warmer",
        "brightness_shift": "darker_but_vibrant",
        "atmospheric_effects": {
            "type": "cyberpunk_complex",
            "glow_colors": [[1.0, 0.4, 0.2], [1.0, 0.6, 0.0]],
            "complex_glows": True,
            "intensity_multiplier": 1.8,
            "saturation_boost": 1.7,
            "coverage_multiplier": 1.4
        }
    }
}
