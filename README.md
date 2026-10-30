# physical-ai-studio: Viam custom training script

Trains an [Intel Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio)
ACT policy from a Viam **sequence dataset**. The resulting Viam model version
contains the OpenVINO export (`act.xml`, `act.bin`, `manifest.json`) and
`viam.json`, which says which cameras and arm feed the policy and how to apply
its actions.

## 1. Record demonstrations

Configure the [capture control sensor](https://docs.viam.com/data/capture-sync/capture-on-demand/)
with:
- each camera on `GetImages`;
- the follower arm on `JointPositions`;
- optionally, a teleop leader arm on `JointPositions`.

Use the same frequency for all of them, e.g. 10 Hz. Call
`start_capture` with `{"tags": ["cmd:<instruction>"]}` at the start of each
demonstration and `stop_capture` at the end. Each pair becomes one sequence, and
each sequence becomes one training episode. Add the sequences to a sequence dataset.

## 2. Custom arguments

| arg | default | meaning |
|---|---|---|
| `arm` | required | follower arm component |
| `cameras` | required | space-separated camera components; the first is the frame clock |
| `action_source` | `next_state` | or a leader arm component whose `JointPositions` is the action (same joint count as `arm`) |
| `policy` | `act` | only `act` for now |
| `fps` | `10` | must match the capture `frequency_hz` |
| `image_size` | unset | downscale the longest image side; also ACT's input size. `224` makes jobs much faster |
| `vcodec` | `auto` | `auto` uses the first of `h264_nvenc`, `h264`, `libsvtav1` that can encode on the machine |
| `workers` | half the cores with `image_size` | conversion processes; otherwise the converter's choice |
| `task` | unset | instruction for sequences without a `cmd:` tag |
| `max_epochs` | `100` | |
| `batch_size` | `8` | |
| `chunk_size` | `100` | actions predicted per inference; 100 is about 10 s at 10 fps |
| `n_action_steps` | `chunk_size` | actions executed before predicting again (≤ `chunk_size`); lower means the policy replans more often |

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

Build a local `<image>` with `docker buildx build --platform linux/amd64 --build-arg CONVERTER_REF=<commit> -t viam-openvino-physicalai:dev --load .`

Tests: `docker run --rm --platform linux/amd64 --shm-size=2g -v "$PWD":/work -w /work -e PYTHONPATH=/work/physical-ai-studio <image> python3 -m pytest tests -q`

## 4. Build and register the container

```sh
docker buildx build --platform linux/amd64 --build-arg CONVERTER_REF=<commit> \
  -t docker.io/<user>/viam-openvino-physicalai:0.3.0-<n> --push .
viam train containers register --uri=docker.io/<user>/viam-openvino-physicalai:0.3.0-<n> \
  --description="physicalai-train 0.3.0" --org-id=<org>
```

`<commit>` is the viam-sequence-to-lerobot commit with `--action-arm`. The Docker Hub repo must be public. Never re-push a tag that's already registered;
bump `<n>` instead.

## 5. Upload and train

```sh
COPYFILE_DISABLE=1 tar --exclude=__pycache__ -czf physical-ai-studio.tar.gz physical-ai-studio/
viam training-script upload --path=physical-ai-studio.tar.gz --org-id=<org> \
  --script-name=physical-ai-studio --version=1 --framework=unspecified
viam train submit custom from-registry --dataset-id=<ds> --org-id=<org> \
  --model-name=<name> --model-version=1 --version=1 \
  --script-name=<org-id>:physical-ai-studio --container-id=<container-id> \
  --args=arm=follower-arm,cameras="'overhead-cam wrist-cam'",image_size=224
viam train logs --job-id=<job-id>
```

Re-uploading the script? Bump `--version` on upload and submit together.

Run the job in the dataset's org: reading another org's images fails with
`OSError: [Errno 5] Input/output error` on `/gcs/...`. Cloud jobs copy the images to local disk with the Cloud Storage transfer manager
before converting; reading them through the `/gcs` mount took hours for 70k images.
