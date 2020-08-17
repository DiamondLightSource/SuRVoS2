
# SuRVoS2 API

survos.api.workspace
          .regions
          .features
          .pipeline
          .annotations
          .render


# SuRVoS2 Commands

## Local Server 

Currently setting server=0:0 uses the hug Local server. See survos2.survos.run_command

python .\bin\survos.py run_server [COMMAND] server=0:0 [PARAMS]

## Remote Server 

python .\bin\survos.py run_server [COMMAND] server=localhost:8123 [PARAMS]

## New viewer

python .\bin\survos.py new_gui

## Classic viewer

python .\bin\survos.py classic_gui
 
## Other commands

python survos.py view_data D:\\datasets\\survos_brain\\ws3\\data.h5


# Workspace commands

Workspaces are stored in CHROOT
They implement a particular folder structure and access protocol for data.
After creation, add_data to a workspace, then create a dataset.


workspace.get_dataset server=0:0 workspace=test_s9 dataset=bob

workspace.list_datasets workspace=test_s19

workspace.add_dataset workspace=test_s9 dataset_name=bob  dtype=float32

workspace.add_data workspace=test_s9 data_fname=D:/datasets/survos_brain/ws3/data.h5  dtype=float32

run_server workspace.add_dataset workspace=test_s3 dataset_name=D:/datasets/survos_brain/ws3/data.h5  dtype=float32

run_server workspace.add_data server=0:0 workspace=test_s11 data_fname=D:/datasets/survos_brain/ws3/data.h5 session='default'

### Groups


# Feature commands

The URIs, src and dst can be either a full path to the data or a name within the current workspace.

__data__ is the default name for the block of data required to be added to a workspace at initialisation with add_data


features.spatial_gradient_3d server=0:0 src=__data__  dst=001_spatial_gradient_3d

features.gaussian_center server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out7.h5

features.simple_laplacian server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out6.h5

features.gaussian_blur server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out7.h5

features.rename workspace=test_s9 feature_id=001_image, new_name=bob

## Region commands

regions.supervoxels server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out1.h5

regions.create workspace=test_s9

regions.get_slice server=localhost:8123 src=D:/datasets/survos_brain/ws3/data.h5 slice_idx=12


# Support for Client-Server operation

python survos.py start_server
Default is to run on port 8123

## URI
URI's for data have the form:

hdf5|h5://
survos://
mrc://


# SuRVoS2 Data model

survos.io
survos.model.dataset
            .workspace
survos.frontend.control.launcher
                       .model
                       .singleton

calling the singleton instance with .g()

Dataset URI looks like:

'survos://default@test_s11:D:\\datasets\\survos_brain\\ws3\\data.h5'
​
Setting dst to 002_gaussian_blur

2020-08-14 09:12:47.534 | INFO     | survos2.frontend.plugins.features:compute_feature:203 - 

Computing features gaussian_blur 

{'src': 'survos://default@test_s11:__data__', 
'dst': 'survos://default@test_s11:features/002_gaussian_blur', 
'modal': True, 'sigma': (1.0, 1.0, 1.0)}

# Segmentation

In order to segment, several things need to be precomputed (supervoxels, filters)
Segmentation requires channels and supervoxels to be calculated.
Refinement requires a segmentation to be computed.

# Entity

An entity is a labeled geometric/vector data is stored in a dataframe with certain columns.
Dataframe has 'z','x','y','class_code'


# Frontend Notes

Napari widget
    Not yet (TODO) updated when local workspace updated. 'View' buttons transfer workspace to viewer.

Feature Plugin
    Modify sets of filters
    Run server-side calculation
    Update local workspace 

Supervoxel Calculation
    Modify supervoxel parameters, choose feature to use
    Run server-side calculation
    Update local workspace

The client and the server are completely separate and communicate via the hug api.

Within the client, PyQT Signals and Slots. 

The application state is in scfg ("survos config).

# Pipeline

A pipeline is to provide a simple, unified interface to several different types of operations.
Ops are currently fufilled by a function that takes and returns a pipeline payload.
This is meant to reflect operations that can occur to the layer stack in the gui or to a workspace on the server.

Example ops are: a layer of points is processed to generate another layer of points. This would be a vector to vector 
operation. Or a layer of points can be processed into a raster mask, a vector-to-raster operation.
A common prediction task involves taking an annotation volume and an image volume and producing a prediction volume,
a (R,R) -> R operation. A detection task takes an image and produce a set of points or other geometry (R->V)

## List of ops

V->V
    Spatial clustering
    Cropping and transformation
V->R
    Mask generation
R->V
    Detection
R->R
    Segmentation

# ROI

Pipeline roi: a pipeline ROI allows for a small region to be run through a pipeline for testing


TODO: Integration of Viewer roi<->workspace roi so
    1) viewer can view a smaller ROI of the workspace
    2) processing (e.g. on server) can be tested on smaller ROI


TODO: An example of the client having a Thumbnail or slice result
