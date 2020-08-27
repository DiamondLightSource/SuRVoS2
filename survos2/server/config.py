
import yaml
import pprint
from survos2.helpers import AttrDict

# Config yamls, kept separate for modularity (e.g. may put in separate files)
# get combined into a master app state dict AppState
survos_config_yaml = """
scfg:
  proj: hunt
  preprocessing: {}
  calculate_features: true
  calculate_supervoxels: true
  load_pretrained_classifier: false
  load_annotation: false
  nb_platform: windows
  random_seed_main: 6868842
  marker_size: 10
  refine_lambda: 1.0
  resample_amt: 0.5
  save_output_files: false
  computing:
    chunks: true
    chunk_size: 1024  # 1 Gb
  mscale: 1.0
  logging:
    level: info
  plot_all: false
  roi_crop: [0,2500, 0, 2500, 0, 2500]
  torch_models_fullpath:  ../experiments
"""

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
    feature: simple_laplacian
    params:
        sigma: 2.1
  filter5:
    plugin: features
    feature: gradient  
    params:
        sigma: 3
"""

pipeline_yaml = """
pipeline_cfg:
  pipeline_params: 
    mask_radius: 10
  predict_params:
    clf: ensemble
    type: rf 
    n_estimators: 10
    proj: False
    max_depth: 20
    n_jobs: 1

"""

scfg = AttrDict(yaml.safe_load(survos_config_yaml)['scfg'])
filter_cfg = AttrDict(yaml.safe_load(filter_yaml))
pipeline_cfg = AttrDict(yaml.safe_load(pipeline_yaml))

# attribute access and AppState class to bundle any other application state
# merge config dictionaries to make app state 
scfg = {**scfg, **filter_cfg}
scfg = {**scfg, **pipeline_cfg}
scfg = AttrDict(scfg)


