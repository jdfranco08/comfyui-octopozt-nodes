import numpy as np
import torch
import os

class OctopoztPngToHDRI:
    """
    Convierte una imagen (output de NanaBanana) a formato .exr usando OpenEXR.
    Outputs: image (IMAGE) para SaveImage, exr_path (STRING) para el viewer 3D.
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
    RETURN_NAMES = ("image", "exr_path",)
    FUNCTION = "convert"
    CATEGORY = "Octopozt"
    OUTPUT_NODE = True

    def convert(self, image, filename, exposure=1.0):
        import OpenEXR
        import Imath

        img_np = image[0].cpu().numpy().astype(np.float32)
        img_np = np.clip(img_np * float(exposure), 0, None)

        h, w = img_np.shape[:2]

        output_dir = "/tmp/octopozt_hdri"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.exr")

        header = OpenEXR.Header(w, h)
        float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = {'R': float_chan, 'G': float_chan, 'B': float_chan}

        exr = OpenEXR.OutputFile(output_path, header)
        exr.writePixels({
            'R': img_np[:, :, 0].astype(np.float32).tobytes(),
            'G': img_np[:, :, 1].astype(np.float32).tobytes(),
            'B': img_np[:, :, 2].astype(np.float32).tobytes(),
        })
        exr.close()

        print(f"[OctopoztPngToHDRI] EXR guardado en: {output_path}")
        return (image, output_path,)


NODE_CLASS_MAPPINGS = {
    "OctopoztPngToHDRI": OctopoztPngToHDRI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OctopoztPngToHDRI": "Octopozt PNG → EXR",
}
