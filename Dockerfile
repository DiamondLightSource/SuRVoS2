FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

WORKDIR /app
COPY . .

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install -y dialog apt-utils && \
    apt-get install -y  \
        build-essential \
        python3.9 \
        python3-pip \
        git \
        mesa-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libfontconfig1 \
        libxrender1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxi6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        && apt-get clean

RUN sed -i 's|torch==1.12.1|torch==1.12.1+cu116|g' req.txt && \
    sed -i 's|torchvision==0.13.1|torchvision==0.13.1+cu116|g' req.txt && \
    python3 -m pip install -r req.txt --extra-index-url https://download.pytorch.org/whl/cu116

RUN python3 ./survos2/improc/setup.py build_ext --inplace

RUN mkdir survos2_workbench && sed -i 's|/tmp|/app/survos2_workbench|g' settings.yaml && python3 -m  pip install -e .

ENTRYPOINT ["python3", "-m", "napari"]

RUN apt-get install -y wget gnupg2 apt-transport-https && \
    wget -O - https://xpra.org/gpg.asc | apt-key add - && \
    echo "deb https://xpra.org/ focal main" > /etc/apt/sources.list.d/xpra.list

RUN apt-get update -y && \
    apt-get install -y \
        xpra \
        xvfb \
        xterm \
        sshfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="python3 -m napari"
ENV XPRA_EXIT_WITH_CLIENT="yes"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start="$XPRA_START" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY

ENTRYPOINT []