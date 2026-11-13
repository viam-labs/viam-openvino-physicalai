# syntax=docker/dockerfile:1
# Training environment for the physical-ai-studio Viam custom training script.
# Viam ignores ENTRYPOINT/CMD but keeps ENV; it needs bash, python3 and pip on PATH.
# Torch wheels bundle the CUDA runtime; the host driver is mounted by the NVIDIA runtime.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG PHYSICALAI_VERSION=0.3.0
# Commit of viam-labs/viam-sequence-to-lerobot with --action-arm and --grippers.
ARG CONVERTER_REF

# torchcodec's cu128 build dlopens libpython3.12 and NPP (libnppicc.so.12), which torch doesn't ship.
# NVIDIA_DISABLE_REQUIRE: skip the base image's driver-branch check; torch bundles its CUDA runtime.
# /usr/local/nvidia: Google's GPU nodes (Viam cloud training) mount the driver there.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NVIDIA_DISABLE_REQUIRE=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/opt/venv/lib/python3.12/site-packages/nvidia/npp/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libpython3.12t64 ffmpeg libgl1 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

RUN test -n "$CONVERTER_REF" \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "physicalai-train[cu128,nncf]==${PHYSICALAI_VERSION}" \
        torch==2.11.0 torchvision==0.26.0 torchcodec==0.11.1 \
        "nvidia-npp-cu12==12.3.3.*" \
        "viam-sequence-to-lerobot @ git+https://github.com/viam-labs/viam-sequence-to-lerobot@${CONVERTER_REF}" \
        "google-cloud-storage>=2.14" \
        pytest

# ACT's ResNet18 backbone weights, so jobs don't download them on GPU time.
RUN python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" \
    && python3 -c "import torchcodec.decoders"  # fails the build if libtorchcodec cannot load

# Pi0.5 tokenizes task text with PaliGemma's tokenizer, which is gated on Hugging Face. Cache it
# at the revision physicalai pins so jobs (which have no token) never ask the gated repo.
# Jobs also download checkpoints (lerobot/pi05_base, ~15 GB) into this cache, so it is writable.
# Build with: --secret id=hf_token,src=$HOME/.cache/huggingface/token
ENV HF_HOME=/opt/hf
ARG PALIGEMMA_REVISION=35e4f46485b4d07967e7e9935bc3786aad50687c
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python3 -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', revision='${PALIGEMMA_REVISION}', use_fast=True, padding_side='right')"
# Load it again through physicalai itself, as a job does (no token, network on): a file missing
# from the cache, or a PALIGEMMA_REVISION that no longer matches physicalai's pin after a version
# bump, would make it ask the gated repo and fail the build with a 401.
RUN python3 -c "from physicalai.policies.pi05.preprocessor import Pi05Preprocessor; Pi05Preprocessor().tokenizer" \
    && chmod -R a+rwX /opt/hf
COPY container/GEMMA_TERMS.txt /opt/licenses/GEMMA_TERMS.txt
