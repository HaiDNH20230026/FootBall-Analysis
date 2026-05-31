"""football_analysis — nhận diện & phân tích cầu thủ trên sân (player/ball/GK/referee)."""
from .detection import load_models, load_models_local
from .pipeline import run_video, run_image

__all__ = ["load_models", "load_models_local", "run_video", "run_image"]
__version__ = "0.1.0"