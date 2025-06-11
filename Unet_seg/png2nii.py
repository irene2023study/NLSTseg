import os
import numpy as np
import nibabel as nib
import imageio

# 設定 PNG 圖片和原始 CT 檔案的路徑
Patient_path = r'210218\210218'
png_dir =  os.path.join(Patient_path, 'predict')
ct_path = r'210218\210218_CT.nii.gz'
output_path = Patient_path + '_tumor.nii.gz'

# 讀取原始 CT 影像
ct_image = nib.load(ct_path)
ct_data = ct_image.get_fdata()
ct_affine = ct_image.affine

# 讀取所有的 PNG 圖片並堆疊成一個 3D 數組
png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))


# 確保至少有一張 PNG 圖片
if not png_files:
    raise ValueError("No PNG files found in the specified directory.")

slices = []
for file in png_files:
    img = imageio.imread(os.path.join(png_dir, file))
    if len(img.shape) == 3:  # 如果圖片有 RGB 通道，轉為灰度圖像
        img = img[:, :, 0]
    # 確保圖片是二元的（0 和 255）
    img = (img > 128).astype(np.uint8)  # 將像素值轉換為 0 和 1
    slices.append(img)

# 將堆疊的 2D 圖片轉換為 3D 數組
volume = np.stack(slices, axis=0)  # 確保順序為 (slice, height, width)
volume = np.moveaxis(volume, 0, -1)  # 移動軸以確保為 (height, width, slice)
volume = np.moveaxis(volume, 0, 1)  # 移動軸以確保為 (height, width, slice)

# 檢查數據類型並確保其為整數型
volume = volume.astype(np.uint8)

# 創建 NIfTI 影像，使用原始 CT 的頭部信息
nii_image = nib.Nifti1Image(volume, ct_affine)

# 保存 NIfTI 檔案
nib.save(nii_image, output_path)
