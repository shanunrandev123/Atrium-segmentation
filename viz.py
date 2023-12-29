#%%
import numpy as np
# %%
import pandas as pd

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
# %%
from pathlib import Path
from celluloid import Camera
from IPython.display import HTML
from tqdm import tqdm
# %%
root = Path("C:/Users/Asus/OneDrive/Desktop/atrium/Task02_Heart/imagesTr/")
label = Path("C:/Users/Asus/OneDrive/Desktop/atrium/Task02_Heart/labelsTr/")
# %%

def change_img_to_label_path(path):
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)


# %%
sample_path = list(root.glob("la*"))[0]

# %%
sample_path_label = change_img_to_label_path(sample_path)
# %%

data = nib.load(sample_path)
label = nib.load(sample_path_label)


# %%
mri = data.get_fdata()

# %%
mask = label.get_fdata().astype(np.uint8)
# %%
nib.aff2axcodes(data.affine)
# %%
def normalize(full_vol):
    mu = full_vol.mean()
    std = np.std(full_vol)
    normalized = (full_vol - mu) / std
    return normalized


def standardize(normalized):
    standardized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return standardized


# %%
all_files = list(root.glob("la*"))
len(all_files)
# %%

save_root = Path("Preprocessed")

for counter, path_to_mri_data in enumerate(tqdm(all_files)):
    path_to_label = change_img_to_label_path(path_to_mri_data)
    mri = nib.load(path_to_mri_data)
    assert nib.aff2axcodes(mri.affine) == ("R", "A", "S")
    mri_data = mri.get_fdata()
    label_data = nib.load(path_to_label).get_fdata().astype(np.uint8)
    
    mri_data = mri_data[32:-32, 32:-32]
    label_data = label_data[32:-32, 32:-32]
    
    normalized_mri_data = normalize(mri_data)
    standardized_mri_data = standardize(normalized_mri_data)
    
    if counter < 17:
        curr_path = save_root/"train"/str(counter)
    else:
        curr_path = save_root/"val"/str(counter)
        
    for i in range(standardized_mri_data.shape[-1]):
        slice = standardized_mri_data[:, :, i]
        mask = label_data[:, :, i]
        slice_path = curr_path/"data"
        mask_path = curr_path/"masks"
        
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)
        
        np.save(slice_path/str(i), slice)
        np.save(mask_path/str(i), mask)
        
        
        
        
    
        


# %%

path = Path("Preprocessed/train/0")

file = "50.npy"

slice = np.load(path/"data"/file)

mask = np.load(path/"masks"/file)


# %%

plt.figure()

plt.imshow(slice, cmap='bone')
mask = np.ma.masked_where(mask == 0, mask)
plt.imshow(mask, alpha=0.5)
# %%

# %%
