import yaml
import pprint
from survos2.helpers import AttrDict


survos_config_yaml = """
cfg:
  proj: unnamed
  preprocessing: {}
  random_seed_main: 32
  marker_size: 10
  save_output_files: false
  plot_all: false
  torch_models_fullpath:  ../experiments
  current_annotation: 001_level
"""

# Filters (features and sr)
filter_yaml = """
filter_cfg:
  superregions1:
    plugin: superregions
    feature: slic
    params:
      slic_feat_idx: -1
      compactness: 20
      postprocess: false
      sp_shape:
      - 18
      - 18
      - 18
  filter1: 
    plugin: features
    feature: gaussian
    params:
        sigma: 2
  filter2: 
    plugin: features
    feature: gaussian
    params:
        sigma: 2
  filter3:
    plugin: features
    feature: tvdenoising3d
    params:
        lamda: 3.7
  filter4:
    plugin: features
    feature: laplacian
    params:
        kernel_size: 3
  filter5:
    plugin: features
    feature: gradient  
    params:
        sigma: 3
"""

# segmentation pipeline
# saved models
pipeline_yaml = """
pipeline:
  calculate_features: true
  calculate_supervoxels: true
  load_pretrained_classifier: true
  load_annotation: false
  predict_params:
    clf: ensemble
    type: rf 
    n_estimators: 10
    proj: False
    max_depth: 20
    n_jobs: 1
  mask_params:
    mask_radius: 10      

"""

cfg = AttrDict(yaml.safe_load(survos_config_yaml)["cfg"])
filter_cfg = AttrDict(yaml.safe_load(filter_yaml))
pipeline_cfg = AttrDict(yaml.safe_load(pipeline_yaml))


cfg = {**cfg, **filter_cfg}
cfg = {**cfg, **pipeline_cfg}
cfg = AttrDict(cfg)
