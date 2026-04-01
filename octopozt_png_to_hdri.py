import numpy as np
import torch
import os

class OctopoztPngToHDRI:
    """
    Convierte una imagen (output de NanaBanana) a formato HDRI (.hdr o .exr).
    Outputs: hdri_path (STRING) para usar como ruta, e image (IMAGE) para
    conectar directo a SaveImage u otros nodos de imagen.
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

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "hdri_path",)
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
        img_np_hdr = img_np * exposure

        # Asegurar carpeta de output
        output_dir = "/tmp/octopozt_hdri"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.{format}")

        if format == "hdr":
            iio.imwrite(output_path, img_np_hdr, extension=".hdr")
        elif format == "exr":
            try:
                import OpenEXR
                import Imath
                r = img_np_hdr[:, :, 0].tobytes()
                g = img_np_hdr[:, :, 1].tobytes()
                b = img_np_hdr[:, :, 2].tobytes()
                h, w = img_np_hdr.shape[:2]
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
                # Fallback a HDR si no hay OpenEXR
                output_path = output_path.replace(".exr", ".hdr")
                iio.imwrite(output_path, img_np_hdr, extension=".hdr")

        print(f"[OctopoztPngToHDRI] Guardado en: {output_path}")

        # Devolver el tensor original (sin exposure) para conectar a SaveImage
        return (image, output_path,)


NODE_CLASS_MAPPINGS = {
    "OctopoztPngToHDRI": OctopoztPngToHDRI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OctopoztPngToHDRI": "Octopozt PNG → HDRI",
}
