import numpy as np
import torch
import os
import struct
import math

class OctopoztPngToHDRI:
    """
    Convierte una imagen (output de NanaBanana) a formato Radiance HDR (.hdr).
    Outputs: image (IMAGE) para SaveImage, hdri_path (STRING) para el viewer 3D.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "entorno_octopozt"}),
            },
            "optional": {
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "hdri_path",)
    FUNCTION = "convert"
    CATEGORY = "Octopozt"
    OUTPUT_NODE = True

    def write_hdr(self, path, image):
        """Escribe un archivo Radiance HDR (.hdr) sin usar OpenCV."""
        h, w, c = image.shape
        with open(path, 'wb') as f:
            f.write(b'#?RADIANCE\n')
            f.write(b'FORMAT=32-bit_rle_rgbe\n')
            f.write(b'EXPOSURE=1.0\n')
            f.write(b'\n')
            f.write(f'-Y {h} +X {w}\n'.encode())
            for row in range(h):
                for col in range(w):
                    r, g, b = float(image[row, col, 0]), float(image[row, col, 1]), float(image[row, col, 2])
                    max_val = max(r, g, b)
                    if max_val < 1e-32:
                        f.write(struct.pack('BBBB', 0, 0, 0, 0))
                    else:
                        exp = math.floor(math.log2(max_val)) + 1
                        scale = 2.0 ** (-exp) * 256.0
                        rv = min(255, int(r * scale))
                        gv = min(255, int(g * scale))
                        bv = min(255, int(b * scale))
                        ev = int(exp) + 128
                        f.write(struct.pack('BBBB', rv, gv, bv, ev))

    def convert(self, image, filename, exposure=1.0):
        img_np = image[0].cpu().numpy().astype(np.float32)
        img_hdr = np.clip(img_np * float(exposure), 0, None)
        output_dir = "/tmp/octopozt_hdri"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.hdr")
        self.write_hdr(output_path, img_hdr)
        print(f"[OctopoztPngToHDRI] Guardado en: {output_path}")
        return (image, output_path,)


NODE_CLASS_MAPPINGS = {
    "OctopoztPngToHDRI": OctopoztPngToHDRI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OctopoztPngToHDRI": "Octopozt PNG → HDRI",
}
