import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from Read_LDR_Stack import Read_LDR_Stack
from Read_LDR_Stack_Info import Read_LDR_Stack_Info
from Debevec_CRF import Debevec_CRF
from Build_HDR import Build_HDR
from Reinhard_TMO import Reinhard_TMO
from Gamma_TMO import Gamma_TMO
from Save_HDR import Save_HDR

dir_name = r"D:\UTEC\CICLOV\Computación Gráfica\HDR\HDRToolbox\demos\stack"
file_format = "jpg"
output_folder = r"D:\UTEC\CICLOV\Computación Gráfica\HDR\output"

os.makedirs(output_folder, exist_ok=True)

print("Paso 1) Lectura de las imágenes LDR")
stack, norm_value = Read_LDR_Stack(dir_name, file_format, 1)

# Imprimir las primeras imágenes del stack
print("Imágenes del stack:")
plt.figure(figsize=(10, 5))
for i in range(min(5, stack.shape[3])):
    plt.subplot(1, 5, i + 1)
    plt.imshow(stack[:, :, :, i])
    plt.axis('off')
plt.show()

print("Paso 2) Leer la exposición de las imágenes a partir del exif")
stack_exposure = Read_LDR_Stack_Info(dir_name, file_format)

# Convertir stack_exposure a un array de NumPy
stack_exposure = np.array(stack_exposure)

# Verificar valores de exposición antes de construir el HDR
print("Valores de exposición del stack:", stack_exposure)

# Eliminar imágenes con valores de exposición duplicados
unique_exposures, indices = np.unique(stack_exposure, return_index=True)
if len(unique_exposures) != len(stack_exposure):
    print("Encontrados valores de exposición duplicados. Eliminando duplicados...")
    stack = stack[:, :, :, indices]
    stack_exposure = stack_exposure[indices]

print("Paso 3) Estimar el Camera Response Function (CRF)")
lin_fun, _ = Debevec_CRF(stack, stack_exposure, 256)

# Verificar el CRF
print("CRF (lin_fun):", lin_fun)

plt.figure()
plt.title('The Camera Response Function (CRF)')
plt.plot(range(256), lin_fun[:,0], 'r', range(256), lin_fun[:,1], 'g', range(256), lin_fun[:,2], 'b')
plt.show()

print("Paso 4) Construyendo el mapa de radianza (usando las imágenes y su exposición)")
imgHDR, _ = Build_HDR(stack, stack_exposure, 'LUT', lin_fun, 'Deb97', 'log')

# Verificar los valores de imgHDR
print("Valores de imgHDR:", imgHDR)
print("Shape de imgHDR:", imgHDR.shape)

# Escalar los valores de imgHDR
imgHDR = imgHDR / np.max(imgHDR) * 255.0
print("Valores mínimos y máximos de imgHDR después del escalado:", np.min(imgHDR), np.max(imgHDR))

print("Paso 5) Guardando el mapa de radianza con el formato .hdr")
Save_HDR("output_hdr_image.hdr", imgHDR)

print("Paso 6) Mostrando la versión con tone mapping del mapa de radianza (con gamma encoding)")
imgTMO, _, _ = Reinhard_TMO(imgHDR, 0.18)
# Normalizar imgTMO a [0, 1] antes de aplicar Gamma_TMO
imgTMO = imgTMO / np.max(imgTMO)
imgTMO = Gamma_TMO(imgTMO, 2.2, 0, False)

# Verificar los valores de imgTMO después de la corrección gamma
print("Valores mínimos y máximos de imgTMO después de Gamma_TMO:", np.min(imgTMO), np.max(imgTMO))

cv2.imshow("Tone Mapped Image", imgTMO)
cv2.waitKey(0)
cv2.destroyAllWindows()
