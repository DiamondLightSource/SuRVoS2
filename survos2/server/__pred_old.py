
def make_prediction2(features_stack, annotation_volume, supervoxel_vol, predict_params):
    """Prepare superregions and predict
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        supervoxel_vol {image volume} -- Labeled image defining superregions.
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=18)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    
    #proj = PCA(n_components='mle', whiten=True, random_state=42)
    #proj = StandardScaler()
    proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=42)
    rnd = 42
    #proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)

    X_train = proj.fit_transform(X_train)
    
    print(X_train)
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, 
                n_estimators=55 ,
                project=predict_params['proj'])

    logger.debug(f"clf: {clf}")

    
    try:
        P = predict(X_train, clf, label=True, probs=True )#, proj=predict_params['proj'])
        probs = P['probs']
        prob_map = invrmap(P['class'], supervoxel_vol)
        num_supervox = supervoxel_vol.max() + 1
        class_labels = P['class'] #- 1
        
        pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
        conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)

        conf_map = invrmap(P['probs'], supervoxel_vol)

        #full_svmask = np.zeros(num_supervox, np.bool)
        #full_svmask[supervoxel_vol.ravel()] = True 
        logger.debug("Finished prediction.")

        from survos2.server.model import SRPrediction

        srprediction = SRPrediction(prob_map, conf_map, probs)

    except Exception as err:
        logger.error(f"Prediction exception: {err}")
        return 0
    return srpredicton



    
def process_anno_and_predict(features_stack, annotation_volume, supervoxel_vol, predict_params):
    """Main superregion 
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        supervoxel_vol {image volume} -- Labeled image defining superregions.
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=6)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

    logger.debug(f"clf: {clf}")

    try:
        P = predict(supervoxel_features, clf, label=True, probs=True) # proj=predict_params['proj'])
        
        probs = P['probs']
        prob_map = invrmap(P['class'], supervoxel_vol)
        num_supervox = supervoxel_vol.max() + 1
        class_labels = P['class'] #- 1
        
        pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
        conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)

        conf_map = invrmap(P['probs'], supervoxel_vol)

        full_svmask = np.zeros(num_supervox, np.bool)
        full_svmask[supervoxel_vol.ravel()] = True 

    except Exception as err:
        logger.error(f"Prediction exception: {err}")

    return prob_map, probs, pred_map, conf_map, P

