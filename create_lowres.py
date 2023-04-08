import cv2
import os
import sys

def img_resize(img_file, shape=(112, 112)):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, shape)
    return img

base_path = "D:\\Kuliah\\Skripsi\\lfw"
path_112 = "lfw-112x112"
path_64 = "lfw-64x64"
path_32 = "lfw-32x32"
path_16 = "lfw-16x16"

# Resize to 64x64
print("Resizing to 64x64...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (64, 64))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_64, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\r Progress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("Resize to 64x64 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)

# Resize to 32x32
print("Resizing to 32x32...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (32, 32))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_32, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\r Progress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("Resize to 32x32 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)

# Resize to 16x16
print("Resizing to 16x16...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (16, 16))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_16, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\r Progress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("Resize to 16x16 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)