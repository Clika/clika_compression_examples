FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG CC_LICENSE_KEY
ENV CC_LICENSE_KEY=${CC_LICENSE_KEY}

ARG USER_NAME="root"
ARG CLIKA_LICENSE_FILE_DIRECTORY="/${USER_NAME}/.clika"
ARG CLIKA_LICENSE_FILE_PATH="${CLIKA_LICENSE_FILE_DIRECTORY}/.cc_license"

RUN if [ -z ${CC_LICENSE_KEY} ]; then echo "CC_LICENSE_KEY is empty! please provide a CC_LICENSE_KEY"; exit 1; fi


# install dependencies for OpenCV which is required for some of the examples
# it is NOT a requirement for `clika-compression` itself
RUN apt update && DEBIAN_FRONTEND=noninter apt  install ffmpeg libsm6 libxext6  -y
RUN apt install -y git # install git to be able to clone required repositories

# creates a file containing the license key so it could be available to the `clika-compression` package
# replaces the interactive `clika-init-license` command (see docs.clika.io installation notes)
RUN mkdir $CLIKA_LICENSE_FILE_DIRECTORY
RUN echo ${CC_LICENSE_KEY} >> $CLIKA_LICENSE_FILE_PATH
# install CLIKA Compression package
RUN pip install "clika-compression" --extra-index-url  \
    https://license:${CC_LICENSE_KEY}@license.clika.io/simple
# validating a proper clika-compression version is installed
RUN if [ "$(pip freeze | grep clika-compression)" = "clika-compression==0.0.0" ]; then echo "CLIKA Compression was \
     not installed properly. Please provide a valid license key"; exit 1; fi
