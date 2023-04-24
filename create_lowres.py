import cv2
import os
import sys

def img_resize(img_file, shape=(112, 112)):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, shape)
    return img

base_path = "/content/lfw"
path_112 = "lfw-112x112"
path_56 = "lfw-56x56"
path_28 = "lfw-28x28"
path_14 = "lfw-14x14"

# Resize to 56x56
print("Resizing to 56x56...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (56, 56))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_56, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\rProgress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("\nResize to 56x56 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)

# Resize to 28x28
print("Resizing to 28x28...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (28, 28))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_28, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\rProgress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("\nResize to 28x28 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)

# Resize to 14x14
print("Resizing to 14x14...")
error_list = []
# Looping untuk setiap subfolder dan gambar dalam folder asli
for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
    total = len(files)
    for i, file in enumerate(files):
        # Memeriksa apakah file adalah gambar
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            try:
                # Normalisasi gambar
                img = img_resize(os.path.join(subdir, file), (14, 14))
                # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                new_folder = os.path.join(base_path, path_14, os.path.basename(subdir))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                cv2.imwrite(os.path.join(new_folder, file), img)
                sys.stdout.write("\rProgress Done: {0}/{1}".format(i+1, total))
                sys.stdout.flush()
            except Exception as e:
                # print(e)
                error_list.append(f"{e} : {os.path.join(new_folder, file)}")
print("\nResize to 14x14 Done!")
print(f"Error Data: {len(error_list)}")
print(error_list)