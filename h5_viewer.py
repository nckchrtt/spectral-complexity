import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILEPATH = "YOUR_PROCESSED_H5_FILE_PATH_HERE.h5"
FRAME_NUMBER = 0

def load_and_view_h5(filepath, frame_idx):
    print(f"Loading dataset: {filepath}")
    
    datasets = {}
    with h5py.File(filepath, 'r') as h5:
        grids = list(h5['/HDFEOS/GRIDS'].keys())
        if not grids:
            print("Error: No grids found in /HDFEOS/GRIDS/")
            return
            
        grid_name = grids[0]
        print(f"Auto-detected grid: {grid_name}")
        
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        if base_fields_path not in h5:
            print(f"Error: Data Fields not found at {base_fields_path}")
            return
            
        data_grp = h5[base_fields_path]
        
        print("Caching datasets into memory...")
        for dataset_name in data_grp.keys():
            # Cache all datasets into memory
            datasets[dataset_name] = data_grp[dataset_name][()]
            print(f"  - Loaded {dataset_name} (Shape: {datasets[dataset_name].shape})")

        
    ortho_img = datasets["ortho_visual"][frame_idx]
    vol_img = datasets["sliding_volume_map"][frame_idx]
    
    # ortho_img is defined as (RGB or RGBAlpha, height, width) in the stacker scripts. 
    # Must transpose to (height, width, RGB) for matplotlib visualization.
    if ortho_img.ndim == 3 and (ortho_img.shape[0] == 4 or ortho_img.shape[0] == 3):
        ortho_img = np.transpose(ortho_img, (1, 2, 0))
        
    print(f"Displaying frame {frame_idx} visuals...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(ortho_img)
    axes[0].set_title(f"Ortho Visual (Frame {frame_idx})")
    axes[0].axis('off')
    
    im = axes[1].imshow(vol_img, cmap='viridis')
    axes[1].set_title(f"Sliding Volume Map (Frame {frame_idx})")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_view_h5(DATASET_FILEPATH, FRAME_NUMBER)
