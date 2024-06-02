import os
import glob
import piexif

from PIL import Image
from Avg_Luminance import Avg_Luminance


def Read_LDR_Stack_Info(dir_name, file_format):
    file_pattern = os.path.join(dir_name, f'*.{file_format}')
    list_files = glob.glob(file_pattern)
    n = len(list_files)
    exposure = [1.0] * n

    for i, file_name in enumerate(list_files):
        img_info = {}

        try:
            img = Image.open(file_name)
            img_info = img.info
            img.close()
        except Exception as err:
            print(err)
            try:
                exif_data = piexif.load(file_name)
                img_info = {
                    'DigitalCamera': {
                        'FNumber': exif_data['Exif'].get(piexif.ExifIFD.FNumber, (1.0, 1.0))[0] / (exif_data['Exif'].get(piexif.ExifIFD.FNumber, (1.0, 1.0))[1] or 1.0),
                        'ISOSpeedRatings': exif_data['Exif'].get(piexif.ExifIFD.ISOSpeedRatings, 1.0),
                        'ExposureTime': exif_data['Exif'].get(piexif.ExifIFD.ExposureTime, (1.0, 1.0))[0] / (exif_data['Exif'].get(piexif.ExifIFD.ExposureTime, (1.0, 1.0))[1] or 1.0)
                    }
                }
            except Exception as err:
                print(err)

        # Chequeitos
        img_info.setdefault('DigitalCamera', {})
        camera_info = img_info['DigitalCamera']

        camera_info['ISOSpeedRatings'] = camera_info.get('ISOSpeedRatings', 1.0)
        if camera_info['ISOSpeedRatings'] == 0:
            camera_info['ISOSpeedRatings'] = 1.0
        
        camera_info['ExposureTime'] = camera_info.get('ExposureTime', 1.0)
        if camera_info['ExposureTime'] == 0:
            camera_info['ExposureTime'] = 1.0

        camera_info['FNumber'] = camera_info.get('FNumber', 1.0)
        if camera_info['FNumber'] == 0:
            camera_info['FNumber'] = 1.0

        if img_info:
            if 'DigitalCamera' in img_info:
                exposure_time = camera_info['ExposureTime']
                aperture = camera_info['FNumber']
                iso = camera_info['ISOSpeedRatings']

                _, value = Avg_Luminance(exposure_time, aperture, iso)
                exposure[i] = value
            else:
                print('Warning: The LDR image does not have camera information')
    return exposure