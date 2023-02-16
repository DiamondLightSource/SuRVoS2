# SuRVoS2

Volumetric Image Segmentation

----------------------------------


## Installation

Please follow the instructions in INSTALL.txt

## SuRVoS2 in Docker

### Build Docker image

The `Dockerfile` files allow a Docker image for SuRVoS2 to be build using Ubuntu 20.04 as platform and also CUDA 11.6.2 for GPU support

The `Dockerfile` file also includes the setup of a preconfigured Xpra server.

To build the image of the master branch, run one of the following commands (type the :

```
git clone https://github.com/DiamondLightSource/SuRVoS2.git
cd SuRVoS2
docker build . -t survos2:latest
```

### Run Docker image
To run the former locally builded Docker image run:
```
docker run -it --rm -p 9876:9876 --gpus all -v $(pwd):/survos2_workbench --workdir=/app/survos2_workbench survos2:latest
```
To run a pre-builded Docker image of version `<tag-version>` run:
```
docker run -it --rm -p 9876:9876 --gpus all -v $(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench quay.io/rosalindfranklininstitute/survos2:<tag-version>
```
With this image you don’t need X server running on the host machine. A browser is sufficient!

Once that’s running, open a browser and browse to http://localhost:9876. This will initiate a virtual desktop that will run
napari with the SuRVoS2 plugin.


## SuRVoS2 in Apptainer/Singularity

### Build Apptainer/Singularity image
You may only build an Apptainer/Singularity image using a pre-builded Docker image (of version `<tag-version>`) as the base.

You can do this with the following command, where `<Alias>` can be either `apptainer` or `singularity`.

```
<Alias> build survos2-<tag-version>.sif docker://quay.io/rosalindfranklininstitute/survos2:<tag-version>
```

### Run Apptainer/Singularity image
To run a locally builded Apptainer/Singularity image run (with `<Alias>` being either `apptainer` or `singularity`)
```
<Alias> run --cleanenv --no-home --nv --bind=$(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench survos2-<tag-version>.sif
```
To run a pre-builded Apptainer/Singularity image of version `<tag-version>` run:
```
<Alias> run --cleanenv --no-home --nv --bind=$(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench docker://quay.io/rosalindfranklininstitute/survos2:<tag-version>
```
With this image you don’t need X server running on the host machine. A browser is sufficient!

Once that’s running, open a browser and browse to http://localhost:9876. This will initiate a virtual desktop that will run
napari with the SuRVoS2 plugin.

## Documentation

The [SuRVoS2 Tutorial] .

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"SuRVoS2" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[SuRVoS2 Tutorial]: docs/Survos_Training_Sept_2022.pdf
