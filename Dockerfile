# Training environment for the physical-ai-studio Viam custom training script.
# Viam ignores ENTRYPOINT/CMD but keeps ENV; it needs bash, python3 and pip on PATH.
# Torch wheels bundle the CUDA runtime; the host driver is mounted by the NVIDIA runtime.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG PHYSICALAI_VERSION=0.3.0
# Commit of viam-labs/viam-sequence-to-lerobot that includes --action-arm.
ARG CONVERTER_REF

# NVIDIA_DISABLE_REQUIRE: skip the base image's driver-branch check; torch bundles its CUDA runtime.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NVIDIA_DISABLE_REQUIRE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg libgl1 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

RUN test -n "$CONVERTER_REF" \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "physicalai-train[cu128]==${PHYSICALAI_VERSION}" \
        torch==2.11.0 torchvision==0.26.0 torchcodec==0.11.1 \
        "viam-sequence-to-lerobot @ git+https://github.com/viam-labs/viam-sequence-to-lerobot@${CONVERTER_REF}" \
        pytest

# ACT's ResNet18 backbone weights, so jobs don't download them on GPU time.
RUN python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
