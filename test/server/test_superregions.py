def test_srprediction():
    srprediction = sr_prediction(features.features_stack,
                                patch.image_layers['Annotation'],
                                superregions,
                                params.predict_params)


def test_superregions():
    dataset_feats, filtered_stack = prepare_prediction_features([np.array(base_image.data),])
    superregions = generate_supervoxels(dataset_feats,  filtered_stack, 
                                        scfg.feats_idx, scfg.slic_params)

