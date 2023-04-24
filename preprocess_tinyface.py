from PIL import Image
import os
import sys
import numpy as np

base_path = "/content/tinyface"
new_path = "/content/tinyface_fix"

# read image
def img_resize(image_path, new_width=28, new_height=28):
    # Open image using PIL
    with Image.open(image_path) as image:
        # Calculate new height and width while preserving aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        # Create a new image with the final dimensions and fill the empty space with black
        final_image = Image.new(mode='RGB', size=(new_width, new_height), color='black')
        final_image.paste(resized_image, (0, 0))

        return final_image

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