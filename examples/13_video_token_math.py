'''Show how video token counts blow up.'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import multimodal_utils

if __name__ == "__main__":
    for duration in [10, 60, 300, 1800]:
        for fps in [0.5, 1.0, 4.0]:
            r = multimodal_utils.video_tokens(duration, fps, 576)
            print(f"{duration}s @ {fps} fps: {r['total_tokens']:,} tokens")
