# OpenVINO physicalai for Viam

Two parts: a Viam custom training script (`physical-ai-studio/`) and a Viam module
(`viam-labs:openvino-physicalai`) that serves the trained models. See [Policy module](#policy-module).

## physical-ai-studio: Viam custom training script

Trains an [Intel Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio)
ACT, SmolVLA or Pi0.5 policy from a Viam **sequence dataset**. The resulting Viam model version
contains the OpenVINO export (`act.xml`, `act.bin`, `manifest.json`), or the ONNX export
(`act.onnx`, `act.onnx.data`, `manifest.json`) with `export_format=onnx`, and
`viam.json`, which says which cameras, arms and grippers feed the policy and how to apply
its actions. A SmolVLA model version holds `smolvla.xml`, `smolvla.bin`, `tokenizer.xml`,
`tokenizer.bin` and `manifest.json` in place of the ACT files, plus `viam.json`. A Pi0.5 model version holds
`pi05.xml`, `pi05.bin` (about 6 GB), `tokenizer.xml`, `tokenizer.bin`, `manifest.json` and `viam.json`.

## 1. Record demonstrations

Configure the [capture control sensor](https://docs.viam.com/data/capture-sync/capture-on-demand/)
with:
- each camera on `GetImages`;
- each follower arm on `JointPositions`;
- each gripper on `DoCommand`, with a fixed `docommand_input` whose response holds its position as a number (under `position` unless you set `gripper_key`). Make it a percentage of the gripper's travel (0–100), not raw driver units: the value is trained on unscaled;
- optionally, a teleop leader arm on `JointPositions`.

Use the same frequency for all of them, e.g. 10 Hz. Every frame needs a reading from every component within 50 ms, so more components drop more frames when captures drift. Call
`start_capture` with `{"tags": ["cmd:<instruction>"]}` (or `task:<instruction>`) at the start of each
demonstration and `stop_capture` at the end. Each pair becomes one sequence, and
each sequence becomes one training episode. Add the sequences to a sequence dataset.

Give each task its own instruction, e.g. `cmd:open the box` and `cmd:close the box`. SmolVLA and Pi0.5
learn one policy that follows whichever instruction it is given; ACT ignores the text and
blends every task into one behavior (the training job warns when that happens).

## 2. Custom arguments

| arg | default | meaning |
|---|---|---|
| `arm` | required | space-separated follower arm components; with several, state and action are `[arm1 joints, gripper1, arm2 joints, gripper2]` |
| `grippers` | unset | space-separated grippers captured with `DoCommand`, one per arm in the same order |
| `gripper_key` | `position` | numeric field of each gripper's `DoCommand` response |
| `cameras` | required | space-separated camera components; the first is the frame clock |
| `action_source` | `next_state` | or a leader arm component whose `JointPositions` is the action (same joint count as `arm`); leader arms are single-arm only, without grippers |
| `policy` | `act` | or `smolvla`, a vision-language-action model conditioned on the task text, or `pi05`, Physical Intelligence's Pi0.5 (~3.6B parameters, LoRA fine-tuned) |
| `export_format` | `openvino` | or `onnx`, for targets without OpenVINO; the module runs ONNX on CPU only; smolvla and pi05 are `openvino` only |
| `pretrained` | `lerobot/smolvla_base` / `lerobot/pi05_base` | smolvla/pi05: checkpoint to fine-tune (Hugging Face id or directory). The job downloads it (about 1 GB for `smolvla_base`, 15 GB for `pi05_base`), so it needs Hugging Face access |
| `num_steps` | `10` | smolvla/pi05: denoising steps per inference |
| `lora_rank` | `32` | pi05 only: LoRA adapter rank; 0 fine-tunes every weight, which does not fit a 40 GB GPU |
| `fps` | `10` | must match the capture `frequency_hz` |
| `image_size` | unset | downscale the longest image side; also ACT's input size. `224` makes jobs much faster |
| `vcodec` | `auto` | `auto` uses the first of `h264_nvenc`, `h264`, `libsvtav1` that can encode on the machine |
| `workers` | half the cores with `image_size` | conversion processes; otherwise the converter's choice |
| `task` | unset | instruction for sequences without a `cmd:` or `task:` tag |
| `max_epochs` | `100` | |
| `batch_size` | `8` | per device; a multi-GPU job logs the effective batch (batch_size × GPUs) |
| `devices` | `auto` | GPUs to train on: `auto` uses every visible GPU, with distributed data parallel (DDP) when there are several, which halves the steps per epoch on two; `1` forces one |
| `chunk_size` | `100` | act only: actions predicted per inference; 100 is about 10 s at 10 fps. smolvla's and pi05's are fixed by `pretrained` (50) |
| `n_action_steps` | the chunk size | actions executed before predicting again (≤ the chunk size); lower means the policy replans more often |

## 3. Run locally

```sh
viam dataset export --dataset-id=<id> --destination=export/
cd export && unzip *.zip && cd ..    # parquets must sit next to binary_data/
docker run --rm --platform linux/amd64 --shm-size=2g -v "$PWD":/work -w /work \
  -e PYTHONPATH=/work/physical-ai-studio <image> python3 -m model.training \
  --binary_data_file=export/binary_data.parquet --tabular_data_file=export/tabular_data.parquet \
  --sequences_file=export/sequences.parquet --model_output_directory=out \
  --arm=follower-arm --cameras="overhead-cam wrist-cam" --image_size=224 --max_epochs=1
```

`viam training-script test-local` does not support sequence datasets.

Build a local `<image>` with `docker buildx build --platform linux/amd64 -f container/Dockerfile --build-arg CONVERTER_REF=<commit> --secret id=hf_token,src=$HOME/.cache/huggingface/token -t viam-openvino-physicalai:dev --load .`

Tests: `docker run --rm --platform linux/amd64 --shm-size=2g -v "$PWD":/work -w /work -e PYTHONPATH=/work/physical-ai-studio <image> python3 -m pytest tests/test_training.py -q`

## 4. Build and register the container

```sh
docker buildx build --platform linux/amd64 -f container/Dockerfile --build-arg CONVERTER_REF=<commit> \
  --secret id=hf_token,src=$HOME/.cache/huggingface/token \
  -t docker.io/<user>/viam-openvino-physicalai:0.3.0-<n> --push .
viam train containers register --uri=docker.io/<user>/viam-openvino-physicalai:0.3.0-<n> \
  --description="physicalai-train 0.3.0" --org-id=<org>
```

`<commit>` is the viam-sequence-to-lerobot commit with `--action-arm` and `--grippers`. The Docker Hub repo must be public. Never re-push a tag that's already registered;
bump `<n>` instead.

Building needs a Hugging Face token (from `hf auth login`) whose account has accepted the terms of
`google/paligemma-3b-pt-224`; the image redistributes that tokenizer under the Gemma Terms of Use,
included at `/opt/licenses/GEMMA_TERMS.txt`.

## 5. Upload and train

### Train with the registered script and container

The script is public in the viam-dev org (`e76d1b3b-0468-4efd-bb7f-fb1d2b352fcb`). The container is
registered in the hackathons org (`6d6c7293-bc67-43e4-9c8b-31c76aac27d3`) and is private to it:

| | ID | version |
|---|---|---|
| training script | `e76d1b3b-0468-4efd-bb7f-fb1d2b352fcb:physical-ai-studio` | `9` |
| container | `1045eccc-b3b7-48fe-86cf-5230b82c336d` | `hipsterbrown/viam-openvino-physicalai:0.3.0-5` |

Run the job, and keep the dataset, in the hackathons org:

```sh
viam train submit custom from-registry --dataset-id=<dataset-id> \
  --org-id=6d6c7293-bc67-43e4-9c8b-31c76aac27d3 \
  --model-name=<model-name> --model-version=1 \
  --script-name=e76d1b3b-0468-4efd-bb7f-fb1d2b352fcb:physical-ai-studio --version=9 \
  --container-id=1045eccc-b3b7-48fe-86cf-5230b82c336d \
  --args=arm=follower-arm,cameras="'overhead-cam wrist-cam'",image_size=224
viam train logs --job-id=<job-id>
```

Pick the policy and other options with `--args` (see [Custom arguments](#2-custom-arguments)),
e.g. `--args=arm=xarm,cameras="'realsense-cam webcam'",policy=pi05,fps=30,image_size=224,max_epochs=1`.
Update the IDs here when you upload a new script version or register a new container.

### Upload your own

```sh
COPYFILE_DISABLE=1 tar --exclude=__pycache__ -czf physical-ai-studio.tar.gz physical-ai-studio/
viam training-script upload --path=physical-ai-studio.tar.gz --org-id=<org> \
  --script-name=physical-ai-studio --version=1 --type=robot_policy \
  --framework=registry --module-resource=viam-labs:openvino-physicalai:policy \
  --url=https://github.com/viam-labs/viam-openvino-physicalai
# Once, to describe the script in the registry (update requires --visibility):
viam training-script update --org-id=<org> --script-name=physical-ai-studio --visibility=private \
  --framework=registry --module-resource=viam-labs:openvino-physicalai:policy \
  --description="Train robot policies (ACT, SmolVLA, Pi0.5) on a Viam sequence dataset with Intel physicalai."
viam train submit custom from-registry --dataset-id=<ds> --org-id=<org> \
  --model-name=<name> --model-version=1 --version=1 \
  --script-name=<org-id>:physical-ai-studio --container-id=<container-id> \
  --args=arm=follower-arm,cameras="'overhead-cam wrist-cam'",image_size=224
# Two arms with grippers: quote each list.
#   --args=arm="'left-arm right-arm'",grippers="'left-gripper right-gripper'",cameras="'overhead-cam wrist-cam'"
viam train logs --job-id=<job-id>
```

Re-uploading the script? Bump `--version` on upload and submit together. `--module-resource` needs
`--framework=registry`, which deploys the trained model together with the policy module.

Run the job in the dataset's org: reading another org's images fails with
`OSError: [Errno 5] Input/output error` on `/gcs/...`. Cloud jobs copy the images to local disk with the Cloud Storage transfer manager
before converting; reading them through the `/gcs` mount took hours for 70k images.

## Policy module

The `viam-labs:openvino-physicalai:policy` generic service runs a model trained above with OpenVINO (or onnxruntime for ONNX exports) and
answers the `viam-labs:vla:policy` DoCommand protocol. Pair it with `viam-labs:vla:controller`, which owns
the cameras, the arm, scheduling and safety.

| attribute | default | meaning |
|---|---|---|
| `model_path` | required | directory holding `viam.json` and the OpenVINO or ONNX export |
| `device` | `AUTO` | `AUTO`, `NPU`, `GPU` or `CPU`; ONNX exports accept only `AUTO` or `CPU`; for SmolVLA and Pi0.5 use `GPU` (`CPU` works but takes about 2 s per chunk for SmolVLA and 11 s for Pi0.5; `NPU` is rejected because neither compiles there) |
| `warmup_inferences` | `2` | throwaway inferences after loading |
| `load_timeout_s` | `600` | give up loading after this long |

Deploy the trained model as an ML model package and point `model_path` at it. viam-server replaces
`${packages.ml_model.<package-name>}` with the package's absolute local path before the module sees the config.
The package must be in (or public to) the machine's org.

```json
{
  "packages": [{"name": "box-opener-act", "package": "<org-id>/box-opener-act", "type": "ml_model", "version": "4"}],
  "services": [{
    "name": "policy", "api": "rdk:service:generic", "model": "viam-labs:openvino-physicalai:policy",
    "attributes": {"model_path": "${packages.ml_model.box-opener-act}", "device": "NPU"}
  }]
}
```

Loading runs in the background (compiling for an NPU or GPU can take a while); poll `status` until it is `ready`.

DoCommands (`{"command": ...}`):

| command | returns |
|---|---|
| `status` | `state` (`idle`, `loading`, `ready` or `failed`) and `error` |
| `specs` | the vla specs fields, plus `cameras` (feature key to camera component), `arm` (the first arm) and `components` (each arm's and gripper's `offset`, `dim`, `method`, `apply` and `units` in the state and action vectors, plus a gripper's `key`), taken from `viam.json` |
| `infer` | `actions` and `raw_actions` (the same for ACT) from `images`, `state` and optional `task`, plus `latency_s` (decode and inference) |
| `reset` | clears the model's internal state |

RTC is not supported (`specs.supports_rtc` is false).

### Controller config

Map each key in the policy's `specs.cameras` to its camera component. For the box-opener model:

```json
{
  "policy_service": "policy", "arm": "xarm",
  "cameras": {"images.realsense_cam": "realsense-cam", "images.camera_transform": "camera-transform"},
  "state_joint_indices": [0, 1, 2, 3, 4, 5],
  "task": "open the box"
}
```

For a SmolVLA or Pi0.5 model, pass the instruction per run: `{"command": "start", "task": "close the box"}`
overrides the configured `task`. The package's `viam.json` lists the trained `tasks`.

`state_joint_indices` is `[0 .. state_dim-1]`. Units are degrees, the controller's default.

`viam-labs:vla:controller` drives one arm today; models with several arms or grippers (`specs.components` longer than one) need its bimanual support, which is planned separately.

### Latency

box-opener v4 (384x384 graph inputs) on an Intel Core Ultra X7 358H NUC. "Graph" is the OpenVINO
`benchmark_app` median; "`infer`" is the median `latency_s` over 10 DoCommand calls with two real JPEG frames,
which adds image decoding and InferenceModel's letterbox preprocessing. Other models and input sizes will differ.

| device | graph | `infer` |
|---|---|---|
| GPU | 2.48 ms | 12 ms |
| NPU | 3.79 ms | 15 ms |
| CPU | 83.7 ms | 142 ms |

SmolVLA (`lerobot/smolvla_base`, not fine-tuned, three 256x256 cameras, 10 denoising steps, random
inputs) on the same NUC, median `predict_action_chunk` over 10 calls: GPU 89 ms, CPU 1.9 s. It does
not compile for the NPU.

Pi0.5 (`lerobot/pi05_base`, not fine-tuned, three 224x224 cameras, 10 denoising steps): GPU 230 ms per
chunk, CPU 10.9 s; it does not compile for the NPU.

### Development

```sh
uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/test_wire.py tests/test_policy.py tests/test_build.py -q
```

Set `PHYSICALAI_TEST_MODEL` to a model package directory to run the integration test. In the training
image, `PHYSICALAI_TEST_SMOLVLA=1` and `PHYSICALAI_TEST_PI05=1` turn on the SmolVLA and Pi0.5 train, export
and infer tests: they download `smolvla_base` (about 1 GB) or `pi05_base` (about 15 GB), and Pi0.5 needs a
large GPU or about 60 GB of RAM. Try the module on a
machine with `viam module reload --part-id <part-id>`. To publish, create a GitHub release; the workflow in
`.github/workflows/deploy.yml` builds it in Viam's cloud build.
