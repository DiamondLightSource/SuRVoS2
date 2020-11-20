# Basic startup instructions

Clone environment (or unzip a zipped repo)

Build the extensions: in survos2/improc, run python setup.py build_ext --inplace

Set CHROOT in survos2/config.py or leave as 'tmp'

Create a workspace (requires a new name than existing
workspaces in CHROOT)

Running the server using the Hug Local Server:
python .\survos.py run_server workspace.create server=0:0 workspace=test_s1

Add data to the workspace:

workspace.add_data workspace=test_s1  data_fname=D:/datasets/survos_brain/ws3/data.h5 

(replace the data_fname with an h5 file that has the image data stored as 'data'.)

workspace.add_dataset workspace=test_s1 dataset_name=data  dtype=float32

# Configuration

Inital setup:

Currently, startup config resides in the survos2/config.py, which is where 
the ip address and port of the server are stored.

survos2/settings.yaml should not need to be modified often, and contains the list of plugins that
will be loaded by survos.

The server state is in cfg ("survos config"), which is the master configuration dictionary, broken into
global survos parameters, filter parameters and pipeline (prediction) parameters. 

server/config.py
    server, port
    chroot

Per session setup:
     /projects/[project name]/project.json

start server with a workspace (can be changed)
run_server [command] with workspace param
start gui with a workspace

# Using workspaces

Workspaces are stored in CHROOT

They implement a particular folder structure and access protocol for data.

After creation, add_data to a workspace, then create a dataset.

All work in survos is organized around a workspace. Image processing, supervoxel generation and segmentation
takes place by choosing a group and creating a new dataset in that group, e.g. in the group 'features' with
the dataset name '001_gaussian_blur'.

The settings chosen in the gui are saved as metadata into the dataset.yaml within that datasets folder
(e.g. in workspace_name/session_name/features/001_gaussian_blur/) which sits in that folder along with the data 
in hdf5 format, with a name like chunk_0x0x0.h5)

On opening the workspace on a second occassion, original parameters are loaded from the metadata and can be
modified and the dataset recomputed.

For segmentation operations a pipeline is created that outputs a dataset. Multi-stage segmentation
pipelines with various pre and post processing steps (e.g. morphology and mask generation) can be 
performed.


# Segmentation

In order to segment, several things need to be precomputed (supervoxels, filters)
Segmentation requires channels and supervoxels to be calculated.
Refinement requires a segmentation to be computed.

# Entity

An entity is a labeled geometric/vector data is stored in a dataframe with certain columns.
Dataframe has 'z','x','y','class_code'


# Frontend Notes

The frontend allows interactive segmentation of an image volume by manipulating a workspace.

The classic_gui has a slice viewer that can work over http.

The nu_gui has a both slice and 3d viewers and tools for painting and editing vector geometry layers.
    
* Not yet (TODO) updated when local workspace updated. 'View' buttons transfer workspace to viewer.

Feature Plugin

* Modify sets of filters
* Run calculation, which updates the workspace 
* Use the GUI to load the result into the viewer

Supervoxel Calculation
* Modify supervoxel parameters, choose feature to use
* Run calculation, which updates the workspace 
* Use the GUI to load the result into the viewer
    

The client and the server are completely separate and communicate via the hug api.

Within the client, PyQT Signals and Slots are used. Within the nu_gui, Napari is used for viewing 3d volumes. 


# Plugins

Features

Regions 

Annotations

# Pipeline

A pipeline is to provide a simple, unified interface to several different types of segmentation operations.
Ops are currently fufilled by a function that takes and returns a pipeline payload.
This is meant to reflect operations that can occur to the layer stack in the gui or to a workspace on the server.

Example ops are: a layer of points is processed to generate another layer of points. This would be a vector to vector 
operation. Or a layer of points can be processed into a raster mask, a vector-to-raster operation.
A common prediction task involves taking an annotation volume and an image volume and producing a prediction volume,
a (R,R) -> R operation. A detection task takes an image and produce a set of points or other geometry (R->V)

## List of ops
(V: Vector, R: Raster)

* V->V
    - Spatial clustering
    - Cropping and transformation
* V->R
    - Mask generation
* R->V
    - Detection
* R->R
    - Segmentation

# ROI

Pipeline roi: a pipeline ROI allows for a small region to be run through a pipeline for testing

WIP: Integration of Viewer roi<->workspace roi so

* viewer can view a smaller ROI of the workspace
*  processing (e.g. on server) can be tested on smaller ROI

Using a temp dataset, the user can develop a segmentation pipeline on a ROI and then save it
and apply it to the entire volume.

# Launcher and DataModel

# Datasets


# SuRVoS2 API
```
survos.api.workspace
          .regions
          .features
          .annotations
          .render
          .pipeline
```

# SuRVoS2 Commands

## Local Server 

Currently setting server=0:0 uses the hug Local server. See survos2.survos.run_command

```
python .\bin\survos.py run_server [COMMAND] server=0:0 [PARAMS]
```

## Remote Server 
```
python .\bin\survos.py run_server [COMMAND] server=localhost:8123 [PARAMS]
```
## New viewer

```
python .\bin\survos.py new_gui
```
## Classic viewer
```
python .\bin\survos.py classic_gui
```
## Other commands
```
python survos.py view_data D:\\datasets\\survos_brain\\ws3\\data.h5
```

# Workspace commands


```
workspace.get_dataset server=0:0 workspace=test_s9 dataset=bob

workspace.list_datasets workspace=test_s19

workspace.add_dataset workspace=test_s9 dataset_name=bob  dtype=float32

workspace.add_data workspace=test_s9 data_fname=D:/datasets/survos_brain/ws3/data.h5 

run_server workspace.add_dataset workspace=test_s3 dataset_name=D:/datasets/survos_brain/ws3/data.h5  dtype=float32

run_server workspace.add_data server=0:0 workspace=test_s11 data_fname=D:/datasets/survos_brain/ws3/data.h5 session='default'
```

### Groups


# Feature commands

The URIs, src and dst can be either a full path to the data or a name within the current workspace.

```__data__```
 is the default name for the block of data required to be added to a workspace at initialisation with add_data

```
features.spatial_gradient_3d server=0:0 src=__data__  dst=001_spatial_gradient_3d

features.gaussian_center server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out7.h5

features.simple_laplacian server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out6.h5

features.gaussian_blur server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out7.h5

features.rename workspace=test_s9 feature_id=001_image, new_name=bob

```
## Region commands

```
regions.supervoxels server=0:0 src=D:/datasets/survos_brain/ws3/data.h5  dst=D:/datasets/survos_brain/out1.h5

regions.create workspace=test_s9

regions.get_slice server=localhost:8123 src=D:/datasets/survos_brain/ws3/data.h5 slice_idx=12

```
# Support for Client-Server operation

python survos.py start_server
Default is to run on port 8123

## URI
URI's for data have the form:

```
hdf5|h5://
survos://
mrc://

```
# SuRVoS2 Data model
```
survos.io
survos.model.dataset
            .workspace
survos.frontend.control.launcher
                       .model
                       .singleton
```
calling the singleton instance with .g()

Dataset URI looks like:

```
'survos://default@test_s11:D:\\datasets\\survos_brain\\ws3\\data.h5'
```

Example output showing source and dst

```
Setting dst to 002_gaussian_blur

2020-08-14 09:12:47.534 | INFO     | survos2.frontend.plugins.features:compute_feature:203 - 

Computing features gaussian_blur 

{'src': 'survos://default@test_s11:__data__', 
'dst': 'survos://default@test_s11:features/002_gaussian_blur', 
'modal': False, 'sigma': (1.0, 1.0, 1.0)}
```

modal refers to the use of multiprocessing for background processing (TODO: windows issues with multiprocessing)

