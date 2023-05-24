
# SuRVoS2 in Docker

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
#### Run using a locally built Docker image
To run the former locally built Docker image run:
```
docker run -it --rm -p 9876:9876 --gpus all -v $(pwd):/survos2_workbench --workdir=/app/survos2_workbench survos2:latest
```
#### Run using a published Docker image (Expected soon. Docker images for SuRVoS2 have not been yet published)
To run a published Docker image of version `<tag-version>` from a `<docker-image-registry>` run:
```
docker run -it --rm -p 9876:9876 --gpus all -v $(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench <docker-image-registry>/survos2:<tag-version>
```
With this image you don’t need X server running on the host machine. A browser is sufficient!

Once that’s running, open a browser and browse to http://localhost:9876. This will initiate a virtual desktop that will run
napari with the SuRVoS2 plugin.


## SuRVoS2 in Apptainer/Singularity (Expected soon. Docker images for SuRVoS2 have not been published yet)

### Build Apptainer/Singularity image
You may only build an Apptainer/Singularity image using a published Docker image (of version `<tag-version>`) from a
`<docker-image-registry>` as the base.

You can do this with the following command, where `<Alias>` can be either `apptainer` or `singularity`.

```
<Alias> build survos2-<tag-version>.sif docker://<docker-image-registry>/survos2:<tag-version>
```

### Run Apptainer/Singularity image
#### Run using a locally built Apptainer/Singularity image
To run a locally built Apptainer/Singularity image run (with `<Alias>` being either `apptainer` or `singularity`)
```
<Alias> run --cleanenv --no-home --nv --bind=$(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench survos2-<tag-version>.sif
```
#### Run using a published Docker image
To run a published Apptainer/Singularity image of version `<tag-version>` from a `<docker-image-registry>` run:
```
<Alias> run --cleanenv --no-home --nv --bind=$(pwd):/app/survos2_workbench --workdir=/app/survos2_workbench docker://<docker-image-registry>/survos2:<tag-version>
```
With this image you don’t need X server running on the host machine. A browser is sufficient!

Once that’s running, open a browser and browse to http://localhost:9876. This will initiate a virtual desktop that will run
napari with the SuRVoS2 plugin.