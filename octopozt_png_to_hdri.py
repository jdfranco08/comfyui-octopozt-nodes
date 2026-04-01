import numpy as np
import torch
import os

class OctopoztPngToHDRI:
    """
    Convierte una imagen (output de NanaBanana) a formato HDRI (.hdr o .exr)
    para usarla como entorno 3D en Octopozt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "format": (["hdr", "exr"],),
                "filename": ("STRING", {"default": "entorno_octopozt"}),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hdri_path",)
    FUNCTION = "convert"
    CATEGORY = "Octopozt"
    OUTPUT_NODE = True

    def convert(self, image, format, filename, exposure):
        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio

        # Tensor BHWC → numpy HWC float32
        img_np = image[0].cpu().numpy().astype(np.float32)

        # Aplicar exposure para simular rango HDR
        img_np = img_np * exposure

        # Asegurar carpeta de output
        output_dir = "/tmp/octopozt_hdri"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.{format}")

        if format == "hdr":
            # Formato Radiance HDR — soportado nativamente por imageio
            iio.imwrite(output_path, img_np, extension=".hdr")
        elif format == "exr":
            try:
                import OpenEXR
                import Imath
                # Separar canales RGB
                r = img_np[:, :, 0].tobytes()
                g = img_np[:, :, 1].tobytes()
                b = img_np[:, :, 2].tobytes()
                h, w = img_np.shape[:2]
                header = OpenEXR.Header(w, h)
                header['channels'] = {
                    'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                }
                exr = OpenEXR.OutputFile(output_path, header)
                exr.writePixels({'R': r, 'G': g, 'B': b})
                exr.close()
            except ImportError:
                # Fallback: guardar como HDR si OpenEXR no está disponible
                output_path = output_path.replace(".exr", ".hdr")
                iio.imwrite(output_path, img_np, extension=".hdr")

        print(f"[OctopoztPngToHDRI] Guardado en: {output_path}")
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "OctopoztPngToHDRI": OctopoztPngToHDRI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OctopoztPngToHDRI": "Octopozt PNG → HDRI",
}
