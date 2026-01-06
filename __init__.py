from .anime_upscale_nodes import SnJakeAnimeUpscaleCheckpointLoader, SnJakeAnimeUpscaleInference

NODE_CLASS_MAPPINGS = {
    "SnJakeAnimeUpscaleCheckpointLoader": SnJakeAnimeUpscaleCheckpointLoader,
    "SnJakeAnimeUpscaleInference": SnJakeAnimeUpscaleInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SnJakeAnimeUpscaleCheckpointLoader": "\U0001F60E Anime Upscale Loader",
    "SnJakeAnimeUpscaleInference": "\U0001F60E Anime Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
