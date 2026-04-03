import numpy as np
import warnings
from scipy import ndimage

def get_landsat_mask(data_grp, f_idx, shape, 
                     sun_elevation_threshold=25, 
                     cloud_dilation=0, 
                     qa_reject_mask=0b111111, 
                     radsat_accept_value=0, 
                     aerosol_accept_level='medium'):
    """
    Generates a boolean spatial mask for LANDSAT data using Quality Assessment (QA) bands.
    Valid pixels return True, masked pixels return False.
    
    Args:
        data_grp (h5py.Group): HDF5 group containing the data
        f_idx (int): index of the frame to process
        shape (tuple): shape of the data
        sun_elevation_threshold (float): threshold for sun elevation
        cloud_dilation (int): number of iterations for cloud dilation
        qa_reject_mask (int): mask for QA reject values
        radsat_accept_value (int): value for radsat accept
        aerosol_accept_level (str): level of aerosol accept
    Returns:
        valid_mask (np.ndarray): boolean spatial mask
    """
    # Mapped levels for Aerosol_Optical_Depth
    AEROSOL_DICT = {
        'low': [2, 4, 32, 66, 68, 96, 100],
        'medium': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164],
        'high': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 192, 194, 196, 224, 228] # Aerosol_Optical_Depth > 0.3
    }
    
    valid_mask = np.ones(shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    
    # Sun Elevation Check (Fails loudly if attribute is missing)
    sun_elev_arr = data_grp['surface_reflectance'].attrs['sun_elevation']
    if sun_elev_arr[f_idx] < sun_elevation_threshold:
        return np.zeros(shape, dtype=bool)

    # QA Reject Mask
    qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
    bad_qa_mask = (qa_pixel & qa_reject_mask) != 0
    if cloud_dilation > 0:
        bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~bad_qa_mask

    # RADSAT Accept Value
    bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != radsat_accept_value
    valid_mask &= ~bad_radsat

    # Aerosol Accept Values
    if aerosol_accept_level != 'all':
        aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
        
        accepted_values = AEROSOL_DICT.get(aerosol_accept_level)
        if accepted_values is None:
            raise ValueError(f"Invalid aerosol_accept_level: '{aerosol_accept_level}'. Must be 'low', 'medium', or 'high'.")
            
        invalid_aerosol = ~np.isin(aerosol, accepted_values)
        if cloud_dilation > 0:
            invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=cloud_dilation)
        valid_mask &= ~invalid_aerosol

    return valid_mask

def get_tanager_mask(data_grp, f_idx, shape, 
                     sun_elevation_threshold=25, 
                     cloud_dilation=2, 
                     apply_cloud_mask=True, 
                     uncertainty_threshold=0.1, 
                     aerosol_depth_threshold=0.3):
    """
    Generates a boolean spatial mask for TANAGER data using beta masks and uncertainty.
    Valid pixels return True, masked pixels return False.
    
    Args:
        data_grp (h5py.Group): HDF5 group containing the data
        f_idx (int): index of the frame to process
        shape (tuple): shape of the data
        sun_elevation_threshold (float): threshold for sun elevation
        cloud_dilation (int): number of iterations for cloud dilation
        apply_cloud_mask (bool): whether to apply the cloud mask
        uncertainty_threshold (float): threshold for uncertainty
        aerosol_depth_threshold (float): threshold for aerosol depth
    Returns:
        valid_mask (np.ndarray): boolean spatial mask
    """
    valid_mask = np.ones(shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    
    # Cloud Mask Check
    if apply_cloud_mask:
        c_mask = (data_grp['beta_cloud_mask'][f_idx, ...] == 1)
        cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...] == 1)
        combined_cloud = c_mask | cirrus_mask
        if cloud_dilation > 0:
            combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=cloud_dilation)
        valid_mask &= ~combined_cloud
    
    # Sun Elevation Check (Derived from Sun Zenith)
    zenith = data_grp['sun_zenith'][f_idx, ...]
    valid_mask &= (zenith != -9999.0) & ((90.0 - zenith) >= sun_elevation_threshold)
        
    # Aerosol Optical Depth Check
    aod = data_grp['aerosol_optical_depth'][f_idx, ...]
    bad_aod_mask = (aod == -9999.0) | (aod >= aerosol_depth_threshold) | np.isnan(aod)
    if cloud_dilation > 0:
        bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~bad_aod_mask
        
    # Surface Reflectance Uncertainty Check
    gw_mask = data_grp['surface_reflectance'].attrs['all_good_wavelengths']
    valid_bands = gw_mask[f_idx].astype(bool)
    
    # Suppress all-NaN slice warnings since we explicitly catch the resulting NaNs on the next line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0)
        
    unc_mask = (unc == -9999.0) | (unc >= uncertainty_threshold) | np.isnan(unc)
    if cloud_dilation > 0:
        unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~unc_mask
        
    return valid_mask

def maximumDistance(data, num_endmembers):
    '''
    Args:
        data (np.ndarray): 3D image cube [nPixels0, nPixels1, nbands]
        num_endmembers (int): number of endmembers to be calculated (choose more than expected to find)
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''      
    # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
    image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")

    if np.min(image2D) < 0:
        warnings.warn('Data contains negative values')
        image2D = np.clip(image2D, 0, 2)
    if np.max(image2D) > 1:
        warnings.warn('Data contains values greater than 1')
        image2D = np.clip(image2D, 0, 1)

    # --- NaN Handling ---
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    if np.sum(valid_mask) < num_endmembers:
        print(f"Not enough valid pixels (no NaNs) to find {num_endmembers} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        return np.full((image2D.shape[1], num_endmembers), np.nan), np.full((1, num_endmembers), np.nan)

    valid_data = image2D[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Transpose to [bands, pixels]
    data_t = np.transpose(valid_data)
    num_bands, num_pix = data_t.shape

    # calculate magnitude of all vectors to find min and max
    magnitude = np.linalg.norm(data_t, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num_endmembers])
    endmembers_index = np.zeros([1, num_endmembers], dtype=int)   

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = data_t[:, idx1]
    endmembers[:, 1] = data_t[:, idx2]
    
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    data_proj = data_t.copy()
    identity_matrix = np.identity(num_bands)

    for i in range(2, num_endmembers):
        diff = data_proj[:, idx2:idx2+1] - data_proj[:, idx1:idx1+1]
        pseudo = np.linalg.pinv(diff)
        data_proj = np.matmul((identity_matrix - np.matmul(diff, pseudo)), data_proj)

        idx1 = idx2
        vec = data_proj[:, idx2:idx2+1] 
            
        diff_new = np.sum(np.square(vec - data_proj), axis=0)

        idx2 = np.argmax(diff_new)

        endmembers[:, i] = data_t[:, idx2]
        endmembers_index[0, i] = valid_indices[idx2]

    return endmembers, endmembers_index

def calcGramLocalVolumes(endmembers, localization_vector):
    '''
    Calculates the estimated volume of the parallelotope formed by the endmembers.
    
    Args:
        endmembers (np.ndarray): [nbands, num_endmembers]
        localization_vector (np.ndarray): [nbands]
    Returns:
        volumes (np.ndarray): [num_endmembers]
    '''
    localized_vectors = endmembers - localization_vector[:, np.newaxis]
    gram = np.matmul(localized_vectors.T, localized_vectors)
    
    N = gram.shape[0]
    volumes = np.zeros(N)
    
    for i in range(1, N + 1):
        sub_gram = gram[:i, :i]
        det = np.linalg.det(sub_gram)
        if det < 0:
            det = 0.0
        volumes[i-1] = np.sqrt(det)
        
    return volumes

def process_volume_frame(frame_data, num_endmembers, gram_type, norm_type):
    '''
    Calculates the endmembers, endmember spectra, and volume curves for a set of endmembers determined from a single image.
    
    Args:
        frame_data (np.ndarray): [nbands, height, width]
        num_endmembers (int): number of endmembers to be calculated
        gram_type (str): type of gram matrix to be calculated
        norm_type (str): type of normalization to be applied
    Returns:
        endmembers (np.ndarray): [nbands, num_endmembers]
        endmember_indices (np.ndarray): [1, num_endmembers]
        volume (np.ndarray): [num_endmembers]
    '''
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands))
    if gram_type == 'minEndmember': print("Localizing Gram to second endmember")
    else: print("Localizing Gram to 0")

    if norm_type == 'bandCount': print(f"Normalizing Endmembers by √{bands}")
    else: print("No Endmember Normalization Applied")

    endmembers, endmember_indices = maximumDistance(img, num_endmembers)
    localizationVec = endmembers[:,1]

    if gram_type == 'minEndmember':
        remainingEndmembers = np.delete(endmembers,1,axis=1)
        volume = calcGramLocalVolumes(remainingEndmembers,localizationVec)
        volume = np.insert(volume,0,0.0)
    else:
        volume = calcGramLocalVolumes(endmembers,np.zeros(bands))

    if norm_type == 'bandCount':
        m_array = np.arange(1, len(volume) + 1)
        volume = volume / np.power(bands, (m_array / 2.0))

    return endmembers, endmember_indices, volume

def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, gram_type, norm_type):
    '''
    Implements a sliding window filter approach to calculate the estimated volume of the parallelotope formed by endmembers determined in the window. 
    
    Args:
        frame_data (np.ndarray): [nbands, height, width]
        tile_size (int): size of the sliding tile
        stride (int): stride of the sliding tile
        num_endmembers (int): number of endmembers to be calculated
        gram_type (str): type of gram matrix to be calculated
        norm_type (str): type of normalization to be applied
    Returns:
        volumes (np.ndarray): [num_endmembers]
    '''
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int8)

    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            tile = img[y_start:y_end, x_start:x_end, :]
            endmembers, _ = maximumDistance(tile, num_endmembers)
            localizationVec = endmembers[:,1]

            if gram_type == 'minEndmember':
                remainingEndmembers = np.delete(endmembers,1,axis=1)
                volume = calcGramLocalVolumes(remainingEndmembers,localizationVec)
                volume = np.insert(volume,0,0.0)
            else:
                volume = calcGramLocalVolumes(endmembers,np.zeros(bands))

            if norm_type == 'bandCount':
                m_array = np.arange(1, len(volume) + 1)
                volume = volume / np.power(bands, (m_array / 2.0))

            vol_val = np.max(volume[2:])

            sum_map[y_start:y_end, x_start:x_end] += vol_val
            count_map[y_start:y_end, x_start:x_end] += 1
            
    return sum_map / count_map

def generate_rgba_image(frame_sr, red_idx=3, green_idx=2, blue_idx=1, low=2, high=98, gamma=1.2):
    '''
    Generates an RGBA image from a frame of surface reflectance data.
    
    Args:
        frame_sr (np.ndarray): [nbands, height, width]
        red_idx (int): index of the red band
        green_idx (int): index of the green band
        blue_idx (int): index of the blue band
        low (int): lower percentile for contrast stretching
        high (int): upper percentile for contrast stretching
        gamma (float): gamma correction factor
    Returns:
        rgba_image (np.ndarray): [height, width, 4]
    '''
    bands, height, width = frame_sr.shape

    if np.all(np.isnan(frame_sr)): 
        return np.zeros((height, width, 4), dtype=np.uint8)

    all_zeros_mask = np.all(frame_sr == 0, axis=0)
    has_nan_mask = np.any(np.isnan(frame_sr), axis=0)
    invalid_pixel_mask = all_zeros_mask | has_nan_mask
    valid_mask = ~invalid_pixel_mask

    rgb_indices = [red_idx, green_idx, blue_idx]
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    for i, idx in enumerate(rgb_indices):
        band_data = frame_sr[idx, :, :]
        valid_pixels = band_data[valid_mask]
        
        if valid_pixels.size == 0:
            continue
            
        p_low, p_high = np.percentile(valid_pixels, (low, high))
        
        if p_low < p_high: 
            stretched = np.clip((band_data - p_low) / (p_high - p_low), 0.0, 1.0)
            if gamma != 1.0:
                with np.errstate(invalid='ignore', divide='ignore'):
                    stretched = np.power(stretched, 1.0 / gamma)
                    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
            
            rgb[:, :, i] = stretched

    rgb_8bit = (rgb * 255).astype(np.uint8)
    
    alpha = np.full((height, width), 255, dtype=np.uint8)
    alpha[invalid_pixel_mask] = 0
    
    rgba_8bit = np.dstack((rgb_8bit, alpha))
    return rgba_8bit



def calc_ndvi_frame(frame_data, red_idx=3, nir_idx=4):
    '''
    Calculates the Normalized Difference Vegetation Index (NDVI) for a frame of surface reflectance data.
    
    Args:
        frame_data (np.ndarray): [nbands, height, width]
        red_idx (int): index of the red band
        nir_idx (int): index of the near-infrared band
    Returns:
        ndvi (np.ndarray): [height, width]
    '''
    bands, height, width = frame_data.shape
        
    red = frame_data[red_idx, :, :]
    nir = frame_data[nir_idx, :, :]
    denominator = nir + red
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / denominator
        
    ndvi[np.isinf(ndvi)] = np.nan
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi

def calc_ndbi_frame(frame_data, swir_idx=5, nir_idx=4):
    '''
    Calculates the Normalized Difference Built-up Index (NDBI) for a frame of surface reflectance data.
    
    Args:
        frame_data (np.ndarray): [nbands, height, width]
        swir_idx (int): index of the short-wave infrared band
        nir_idx (int): index of the near-infrared band
    Returns:
        ndbi (np.ndarray): [height, width]
    '''
    bands, height, width = frame_data.shape
        
    swir = frame_data[swir_idx, :, :]
    nir = frame_data[nir_idx, :, :]
    denominator = swir + nir
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndbi = (swir - nir) / denominator
        
    ndbi[np.isinf(ndbi)] = np.nan
    ndbi = np.clip(ndbi, -1.0, 1.0)
    
    return ndbi

def calculate_global_z_score(volume_array, valid_pixel_mask):
    '''
    Calculates the global Z-score for an array of spectral complexity data.
    
    Args:
        volume_array (np.ndarray): [height, width]
        valid_pixel_mask (np.ndarray): [height, width]
    Returns:
        z_scores (np.ndarray): [height, width]
    '''
    print("Calculating global Z-score for frame")
    height, width = volume_array.shape
    z_scores = np.full((height, width), np.nan, dtype=np.float32)
    
    global_valid_mask = volume_array > 0.0
    stats_mask = global_valid_mask & valid_pixel_mask
    
    if not np.any(stats_mask):
        warnings.warn("calculate_global_z_score warning: No radiometrically valid pixels with volume > 0 found. Returning NaNs.")
        return z_scores
        
    stats_vols = volume_array[stats_mask]
    log_stats_vols = np.log(stats_vols)
    
    global_mean = np.mean(log_stats_vols)
    global_std = np.std(log_stats_vols, ddof=1)
    
    if global_std == 0:
        raise ValueError("calculate_global_z_score failed: Global std is exactly zero.")
        
    apply_vols = volume_array[global_valid_mask]
    log_apply_vols = np.log(apply_vols)
    
    z_scores[global_valid_mask] = (log_apply_vols - global_mean) / global_std
    return z_scores
