
"""
@pytest.mark.skip(reason="todo")
def test_srprediction():
    srprediction = _sr_prediction(features.features_stack,
                                patch.image_layers['Annotation'],
                                superregions,
                                params.predict_params)

@pytest.mark.skip(reason="todo")
def test_superregions():
    dataset_feats, filtered_stack = prepare_prediction_features([np.array(base_image.data),])
    superregions = generate_supervoxels(dataset_feats,  filtered_stack, 
                                        cfg.feats_idx, cfg.slic_params)

"""