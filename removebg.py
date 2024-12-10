import os
from rembg import remove
from tqdm import tqdm

# Directory paths
input_dir = 'D:\semester 5\Final Project\\testcamera'     # Ganti dengan direktori gambar yang ingin dihapus background-nya
output_dir = 'D:\semester 5\Final Project\\testcamera'    # Ganti dengan direktori tempat menyimpan hasil gambar

# Buat output directory jika belum ada
os.makedirs(output_dir, exist_ok=True)

# List semua file gambar dalam direktori input
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

# Proses setiap gambar dengan progress bar
for image_file in tqdm(image_files, desc="Processing images", unit="image"):
    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, f"no_bg_{image_file}")

    # Buka gambar dan hapus background
    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()
        output_image = remove(input_image)

    # Simpan hasil ke direktori output
    with open(output_path, 'wb') as output_file:
        output_file.write(output_image)

print(f"Background removal completed for {len(image_files)} images.")
print(f"Images saved to directory: {output_dir}")

