'''Estimate image token counts for VLMs at various resolutions.'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import multimodal_utils

if __name__ == "__main__":
    for w, h in [(224, 224), (336, 336), (448, 448), (1024, 1024)]:
        for patch in [14, 16]:
            r = multimodal_utils.img_tokens(w, h, patch)
            print(f"{w}x{h}, patch {patch}: {r['total_patch_tokens']} tokens")
