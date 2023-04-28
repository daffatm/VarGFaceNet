from PIL import Image
import os
import sys
import numpy as np

base_path = "/content/tinyface_raw"
new_path = "/content/tinyface"

# read image
from PIL import Image

def img_resize(image_path, size=28):
    # buka gambar
    with Image.open(image_path) as img:
        # hitung aspek rasio
        aspect_ratio = img.size[0] / img.size[1]
        
        # tentukan ukuran yang tepat
        if aspect_ratio >= 1:
            new_size = (size, int(size / aspect_ratio))
        else:
            new_size = (int(size * aspect_ratio), size)
        
        # ubah ukuran gambar
        img = img.resize(new_size)
        
        # buat latar belakang hitam
        background = Image.new('RGB', (size, size), (0, 0, 0, 255))
        
        # hitung posisi untuk menempatkan gambar
        offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
        
        # tempatkan gambar di atas latar belakang hitam
        background.paste(img, offset)

        return background

error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(base_path):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                destination_subfolder = os.path.relpath(subdir, base_path)
                new_folder = os.path.join(new_path, destination_subfolder)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                sys.stdout.write("\rProgress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
                img.save(os.path.join(new_folder, file))
                # cv2.imwrite(os.path.join(new_folder, file), img)
            except Exception as e:
                print(e)
                error_list.append(f"{e} : {os.path.join(new_path, file)}")
print("\nResize to 28x28 with padding Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)