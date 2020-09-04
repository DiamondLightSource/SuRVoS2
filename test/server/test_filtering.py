from survos2.server.filtering import gaussian, gaussian_blur, simple_invert

"""



# todo replace with two calls to one function
def crop_and_resample(dataset_in, layers, roi_crop, resample_amt):
    dataset_proc = dataset_in[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3],
                   roi_crop[4]:roi_crop[5]].copy()

    logger.info(f"Prepared region: {dataset_in.shape}")
    dataset_proc = scipy.ndimage.zoom(dataset_proc, resample_amt, order=1)
    logger.info(f"Cropped and resized volume shape: {dataset_proc.shape}")

    layers_proc = []

    for layer in layers:
        # Annotation
        layer_proc = layer[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3], roi_crop[4]:roi_crop[5]]
        layer_proc = scipy.ndimage.zoom(layer_proc, resample_amt, order=1)
        logger.info(f"Cropped and resized layer with shape: {layer_proc.shape}")
        layers_proc.append(layer_proc)

    return dataset_proc, layers_proc

def select_region(imvol : np.ndarray) -> Tuple[float, float, float, float, float, float]:
    vol_shape_z = imvol.shape[0]
    vol_shape_x = imvol.shape[1]
    vol_shape_y = imvol.shape[2]

    zstart, zend = 0, vol_shape_z
    xstart, xend = 0, vol_shape_x
    ystart, yend = 0, vol_shape_y
    return (zstart, zend, xstart, xend, ystart, yend)


def calc_feats(cropped_vol):
    img_vol =cropped_vol
    roi_crop = scfg.roi_crop

    roi_crop = [0,cropped_vol.shape[0],0,cropped_vol.shape[1], 0,cropped_vol.shape[2]]
    
    feature_params = [[gaussian, scfg.filter1['gauss_params']],
    [gaussian, scfg.filter2['gauss_params'] ],
    [simple_laplacian, scfg.filter4['laplacian_params'] ],
    [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]]


    #feature_params = [ 
    #    [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]
    #]

    feats = generate_features(img_vol, feature_params, roi_crop, 1.0)
    
    return feats
    
"""