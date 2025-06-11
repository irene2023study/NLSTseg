import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def rescale_intensity(image, window_center, window_width):
    min_value = window_center - window_width / 2
    newimg = (image - min_value) / float(window_width)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
   
    return newimg

def load_nii(filepath):
    nii_data = nib.load(filepath)
    return nii_data.get_fdata()

def convert_to_HU(image):
    # Assuming CT image, where -1000 is air and 0 is water
    # Modify this if needed based on the actual pixel values in your image
    image_HU = image
    return image_HU

def plot_image(image):
    plt.imshow(image[:, :, 50], cmap='gray')
    plt.axis('off')
    plt.show()

window_center = -500  # Your window center value
window_width = 1400  # Your window width value


nii_file = '203166_CT.nii.gz'
image_data = load_nii(nii_file)
pid = nii_file.split('_')[0]
# Convert to Hounsfield units (HU)
image_HU = convert_to_HU(image_data)
image_truncated = rescale_intensity(image_HU, window_center, window_width)
image_truncated = np.rot90(image_truncated, k=-1)
image_truncated = np.flip(image_truncated, axis=1)


ct_save_path = os.path.join(pid, 'ct')

if not os.path.isdir(ct_save_path):
    os.makedirs(ct_save_path)

# Convert to Hounsfield units (HU)
for s in range(image_truncated.shape[2]):
    ct = Image.fromarray(image_truncated[:, :, s])
    ct.save(os.path.join(ct_save_path, str(s) + '.png'))

    
            