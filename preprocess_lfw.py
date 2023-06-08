import cv2
import os
import sys

base_path = "/content/lfw"
path_112 = "lfw-112x112"
path_56 = "lfw-56x56"
path_28 = "lfw-28x28"
path_14 = "lfw-14x14"

def img_resize(img_file, shape=(112, 112)):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, shape)
    img = cv2.resize(img, (112, 112))
    return img

def do_resize(out_path):
    print(f"Resizing to {out_path}")
    if out_path == path_56:
        img_size = 56
    elif out_path == path_28:
        img_size = 28
    elif out_path == path_14:
        img_size = 14
        
    error_list = []
    # Looping untuk setiap subfolder dan gambar dalam folder asli
    for subdir, dirs, files in os.walk(os.path.join(base_path, path_112)):
        total = len(files)
        for i, file in enumerate(files):
            # Memeriksa apakah file adalah gambar
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                try:
                    # Downscale dan upscale gambar
                    img = img_resize(os.path.join(subdir, file), (img_size, img_size))
                    # Menyimpan gambar yang diubah ukuran dengan nama yang sama di folder baru
                    new_folder = os.path.join(base_path, out_path, os.path.basename(subdir))
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    cv2.imwrite(os.path.join(new_folder, file), img)
                    sys.stdout.write("\rProgress Done: {0}/{1}".format(i+1, total))
                    sys.stdout.flush()
                except Exception as e:
                    # print(e)
                    error_list.append(f"{e} : {os.path.join(new_folder, file)}")
    print(f"\nResize to {out_path} Done!")
    print(f"Error Data: {len(error_list)}")
    print(error_list)
    
if __name__ == '__main__':
    do_resize(path_56)
    do_resize(path_28)
    do_resize(path_14)