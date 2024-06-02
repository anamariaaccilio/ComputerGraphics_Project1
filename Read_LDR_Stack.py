import os
import numpy as np
from PIL import Image
import piexif

def Read_LDR_Stack(dir_name, file_format, bNormalization=0, bToSingle=1):
    if bNormalization:
        bToSingle = True

    norm_value = 1.0

    # Listar todos los archivos del directorio (con el formato dado).
    lista_files = [f for f in os.listdir(dir_name) if f.endswith(f".{file_format}")]
    n = len(lista_files)

    if n > 0:
        img_info = None
        name = os.path.join(dir_name, lista_files[0])

        try:
            img_info = Image.open(name).info
        except Exception as err:
            print("Ocurrió un error abriendo la imagen: ", err)

            try:
                exif = piexif.load(name)
                img_info = {
                    'ColorType': ('truecolor' if (exif['0th'].get(piexif.ImageIFD.SamplesPerPixel, 3) == 3) else 'grayscale'),
                    'BitDepth': round(np.mean(exif['0th'].get(piexif.ImageIFD.BitsPerSample, [8]))),
                    'Width': exif['0th'].get(piexif.ImageIFD.ImageWidth, 0),
                    'Height': exif['0th'].get(piexif.ImageIFD.ImageLength, 0)
                }
            except Exception as err:
                print("No se pudieron cargar la data exif: ", err)
                img_info = {'ColorType': 'grayscale'}  # Establecer 'ColorType' en 'grayscale' como valor predeterminado

        color_channels = 0
        norm_value = 255.0

        if img_info:
            color_channels = img_info.get('NumberOfSamples', 1 if img_info.get('ColorType', 'grayscale') == 'grayscale' else 3)
            bit_depth = img_info.get('BitDepth', 8)

            if img_info.get('ColorType', 'grayscale') == 'grayscale':
                norm_value = 65535.0 if bit_depth == 16 else 255.0
            elif img_info.get('ColorType', 'grayscale') == 'truecolor':
                norm_value = 65535.0 if bit_depth == 48 else 255.0

            width = img_info.get('Width', 0)
            height = img_info.get('Height', 0)

            if width == 0 or height == 0:
                tmp = Image.open(name)
                width, height = tmp.size
                tmp.close()

            # Inicializar el stack con 3 canales por defecto para manejo uniforme
            stack = np.zeros((height, width, 3, n), dtype=np.float32 if bToSingle else np.uint8)

            for i, file_name in enumerate(lista_files):
                print(f"Carga de {file_name} exitosa.")
                img_tmp = Image.open(os.path.join(dir_name, file_name))
                img_array = np.array(img_tmp, dtype=np.float32 if bToSingle else np.uint8)

                # Convertir la imagen a 3 canales si es en escala de grises
                if img_array.ndim == 2:
                    img_array = np.stack((img_array,) * 3, axis=-1)

                stack[:, :, :, i] = img_array

                img_tmp.close()

            if bNormalization:
                stack = stack / norm_value

            return stack, norm_value
        else:
            print("No hay información de las imágenes.")
            return None, norm_value
    else:
        print(f"El directorio no contenía imágenes de formato '.{file_format}'.")
        return None, norm_value
