# FlowTSE-RL

> **Reading note:** This README contains many technical details and has not been
> polished for readability. You can ask an AI agent to read it and help organize
> the relevant setup, configuration, training, and evaluation steps.

Conditional flow matching for target speaker extraction with a UDiT backbone.
The separator starts from Gaussian noise and predicts clean speech features,
conditioned on a noisy mixture and an enrollment utterance. The default model
uses fully bidirectional attention and a frozen BigVGAN decoder.

This is a research implementation inspired by FlowTSE, with a UDiT backbone.
It is not the authors' official FlowTSE implementation. The repository
contains supervised training and tested RL primitives; a complete RL trainer
and reward pipeline are not included.

## Model and recipe

| Recipe | Features | Decoder | Epochs |
| --- | --- | --- | --- |
| `configs/baseline.yaml` | 24 kHz, 100-bin log Mel | Frozen BigVGAN v2 | 500 |

The baseline uses Gaussian-to-clean velocity matching without CFG. The default
conditioning projects `[0, enrollment]` and `[state, mixture]` through a shared
feature projection, then concatenates the two sequences in time. Attention is
fully bidirectional. The UDiT uses width 1024, 16 heads and 17 transformer blocks
(the depth parameter is 16 plus the middle block).

The training recipe uses 3-second random mixture/target crops, 3-second
enrollment, BF16, per-GPU batch size 32, accumulation 1, and AdamW with betas
0.9/0.999. The learning rate warms up for 5 epochs, then follows a cosine from
1e-4 to 1e-5. The reference training uses four GPUs and both noisy Libri2Mix
train-100 and train-360 splits. Validation uses the first 20% of eligible DEV
pairs every 10 epochs.

The shared audio is 16 kHz; the loader resamples it to the model's 24 kHz
frontend. Mixture and target use the same crop and gain, targeting mixture RMS
0.1 with a mixture peak limit of 0.99. Enrollment is a different utterance from
the target speaker, normalized independently to RMS 0.1. Training samples crops and enrollment candidates
randomly; evaluation uses a fixed seed per target pair. Short enrollment uses
its available duration, up to 3 seconds.

## Test-set results

Completed baseline evaluation on noisy Libri2Mix TEST (16 kHz/min), using the
same 6000 target pairs, enrollment choices and seed 42 for every NFE setting.
The 500-epoch training run's checkpoint was selected by DEV PESQ after 460
completed epochs (global step 465060), before this TEST comparison.

Inference uses Gaussian initialization, Euler integration without CFG,
whole-utterance Mel analysis, 3-second Flow chunks and one whole-utterance
BigVGAN decode. The reference evaluation uses BF16 for DiT and FP32 for the
frontend/BigVGAN on one RTX 5090. The portable CLI currently uses FP32, so
reproductions should account for this precision difference.

| NFE | SI-SDR (dB) ↑ | PESQ ↑ | ESTOI ↑ | DNSMOS OVRL ↑ | SIG ↑ | BAK ↑ | P.808 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | -21.4106 | 1.9846 | 0.7669 | 3.2614 | 3.5530 | 4.0356 | 3.6550 |
| 8 | -22.1315 | 1.9095 | 0.7580 | 3.2614 | 3.5635 | 4.0147 | 3.7044 |
| 16 | -22.6316 | 1.8548 | 0.7508 | 3.2502 | 3.5604 | 3.9968 | 3.7248 |
| 32 | -22.9304 | 1.8179 | 0.7448 | 3.2398 | 3.5551 | 3.9847 | 3.7269 |

| NFE | WER (%) ↓ | CER (%) ↓ | WavLM similarity ↑ | WeSpeaker similarity ↑ | Inference time (s) ↓ | RTF ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 16.3679 | 10.3304 | 0.9638 | 0.8834 | 959.16 | 0.0318 |
| 8 | 18.0059 | 11.2970 | 0.9647 | 0.8853 | 1782.37 | 0.0591 |
| 16 | 18.9253 | 11.8559 | 0.9643 | 0.8842 | 3009.55 | 0.0998 |
| 32 | 19.2690 | 12.1195 | 0.9636 | 0.8821 | 5255.45 | 0.1743 |

Each quality/similarity metric covers 6000 pairs. WER/CER cover the same 3004
complete-shorter-target pairs; the other 2996 pairs are excluded from ASR only.
All measured metrics have zero failures. Inference time includes frontend,
Flow, BigVGAN and transfer to CPU, and excludes warmup, data reads, file writes
and scoring. RTF divides this time by 30147.56 seconds of target audio.
Full-precision values are in [the benchmark CSV](benchmarks/librimix_test_nfe.csv);
WER/CER are stored as fractions in the CSV and shown as percentages above.

**Recommended for subsequent RL experiments: NFE=4.** RL repeatedly generates
audio for rollouts and rewards; reducing the number of integration steps saves
separator inference time. With Euler and no CFG, this uses four backbone
evaluations per chunk, one quarter of the calls at NFE=16. The measured
separator inference above is about 3.14 times faster than NFE=16; total RL
speed also depends on decoding, reward scoring and optimization.

In this comparison, NFE=4 also has the highest PESQ/ESTOI and lowest WER/CER;
higher NFE improves P.808, and NFE=8 has the highest speaker similarities.
Keep NFE fixed when comparing RL variants. Use `--steps 4` for the recommended
sampling budget; the SFT configuration's default validation setting remains
16 steps. The examples below select 4 steps explicitly.

## Installation

The reference setup uses two Conda environments on Linux with NVIDIA GPUs.
The training and scoring stacks have different Accelerate and NumPy versions;
install them separately. The Python package supports Python 3.9 or newer, while
the commands below match the reference Python versions.

| Environment | Python | PyTorch / torchaudio | CUDA wheel | Accelerate |
| --- | --- | --- | --- | --- |
| Training / inference | 3.9 | 2.7.1 | cu128 | 0.20.3 |
| Full scoring | 3.10 | 2.8.0 | cu128 | 1.12.0 |

On a fresh Ubuntu/Debian machine, install the audio tools and compiler needed
by PESQ, compiled training kernels and ASR dependencies (or ask the administrator):

```bash
sudo apt-get update
sudo apt-get install -y build-essential libsndfile1 sox libsox-fmt-all zstd rsync
```

A compatible NVIDIA driver must already be installed; check it with
`nvidia-smi`. The following commands use the official CUDA 12.8 PyTorch wheels.
For another platform, use the matching commands from
[PyTorch's version-specific installation guide](https://pytorch.org/get-started/previous-versions/).
Torch and torchaudio versions must match.

Clone the public repository, then create the training environment. Run all
subsequent project commands from this root:

```bash
git clone https://github.com/Jinming00/FlowTSE-RL.git
cd FlowTSE-RL
conda create -n flowtse-train python=3.9 pip -y
conda activate flowtse-train
python -m pip install --upgrade pip 'setuptools==80.9.0' wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/train-constraints.txt -e '.[eval,logging,test]'
python -m pip check
python -m unittest discover -s tests -v
```

[train-constraints.txt](requirements/train-constraints.txt) preserves the key
training versions, including the older Accelerate runtime and its setuptools
compatibility. It is a set of version constraints rather than a complete lock
of every transitive dependency. Scoring-only ASR tests are skipped if their
optional dependencies are absent. The training extra `eval` supplies PESQ and
ESTOI; `logging` supplies optional W&B support. W&B logging is disabled by
default, so training does not require a W&B login. To enable it, follow the
login and configuration steps in [Training and resume](#training-and-resume).

Check GPU access in the training environment:

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

Create the independent scoring environment in the same repository directory:

```bash
conda create -n flowtse-score python=3.10 pip -y
conda activate flowtse-score
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/scoring-constraints.txt -e '.[scoring-gpu,scoring-asr]'
python -m pip check
python score.py --help
```

[scoring-constraints.txt](requirements/scoring-constraints.txt) pins the key
versions used for the scoring checks. For CPU ONNX scoring, install
`'.[scoring,scoring-asr]'` in place of `'.[scoring-gpu,scoring-asr]'` and use
`--scoring-device cpu`. CPU and GPU ONNX Runtime share the same Python module;
choose one package per environment. If ASR is not needed, omit the
`scoring-asr` extra. CPU-only PyTorch wheels can be selected from the official
version guide as well.

The pinned GPU ONNX Runtime uses the CUDA/cuDNN runtime supplied by the CUDA
PyTorch installation. See the
[ONNX Runtime CUDA compatibility guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch).
The DNSMOS CUDA adapter imports PyTorch before creating its ONNX sessions so
those runtime libraries are available. Inspect the scoring environment with:

```bash
python -c "import torch, onnxruntime as ort; print('CUDA available:', torch.cuda.is_available()); print('ORT:', ort.__version__, ort.get_available_providers())"
```

For a CPU-only check, `CPUExecutionProvider` is sufficient. GPU scoring requires
`CUDAExecutionProvider` and a working CUDA device. These versions reflect the existing reference environments; a fresh installation
from these constraints has not been validated.

Data and model weights are separate from Python dependencies. Prepare the
LibriMix CSV/audio layout and BigVGAN files described below. For full scoring,
also supply DNSMOS, Qwen3-ASR, WavLM, WeSpeaker and true-text references using
the paths in the scoring section. `python score.py --metrics full --check-assets`
with those path flags checks the scoring dependencies and asset locations before
scoring. GitHub contains code and aggregate benchmark results. Audio, pairing
CSVs and model weights are distributed separately as described below; account
credentials are not included. The runtime does not download models automatically.

Switch back to `conda activate flowtse-train` for training/inference. Standalone
scoring runs under `conda activate flowtse-score`. To score from a training
process, set `FLOWTSE_SCORING_PYTHON` to the absolute Python executable of the
scoring environment; obtain it with:

```bash
conda run -n flowtse-score python -c "import sys; print(sys.executable)"
```

## Data and pretrained decoder

The default `data.root` is `/share/202611/FlowTSE-RL-share/data`. Dataset paths
in `configs/baseline.yaml` resolve relative to that root. Run commands from the
repository root; paths support environment variables and `~` expansion.

The shared project at `/share/202611/FlowTSE-RL-share/` includes a source
snapshot and the matching noisy Libri2Mix data and pairings under `data/`.
`DATA_READY.json` marks completed delivery; `SHARE_INFO.json` identifies the
source snapshot. Data and pairing files are excluded from GitHub.

| Item | Where to obtain it |
| --- | --- |
| Source and configuration | This repository or the shared source snapshot |
| Audio, enrollment/mixture pairings and ASR references | Shared `data/` directory |
| BigVGAN decoder weights | Download separately using the instructions below |
| Trained separator checkpoint | Shared `checkpoints/baseline-epoch0460-step465060.ckpt`, or produced by your own training |
| Optional scoring model weights | Prepare the assets listed in the evaluation section |

The shared paths below apply to collaborators with access to that filesystem.
For other machines, follow [dataset access without the shared directory](#dataset-access-without-the-shared-directory):
prepare audio through the official LibriMix repository, request the enrollment
pairing files, and set your local data paths. Optional Hugging Face distribution
covers pairing files and the model checkpoint.

| Split | Mixtures | Target pairs | Audio delivery |
| --- | --- | --- | --- |
| train-100 | 13900 | 27800 | Ordinary WAV files |
| train-360 | 50800 | 101600 | Four lossless `.tar.zst` packages |
| dev | 3000 | 6000 | Ordinary WAV files |
| test | 3000 | 6000 | Ordinary WAV files |

The shared layout is:

```text
/share/202611/FlowTSE-RL-share/data/
  LibriMix/Libri2Mix/wav16k/min/
    metadata/mixture_{train-100,train-360,dev,test}_mix_both.csv
    metadata/metrics_{train-100,train-360,dev,test}_mix_both.csv
    train-100/mixture2enrollment.csv
    train-360/mixture2enrollment.csv
    dev/mixture2enrollment.csv
    test/mixture2enrollment.csv
    {train-100,dev,test}/{mix_both,s1,s2,noise}/*.wav
  audio_archives/train-360-{mix_both,s1,s2,noise}.tar.zst
  audio_archives/manifest.json
  asr_references/{dev,test}.json
  licenses/LibriSpeech-LICENSE.txt
```

Use your own writable checkout for experiments. DEV/TEST evaluation can read
the shared directory directly. Full baseline training first requires extracting
train-360 into a writable data directory with at least 95 GiB of free space:

```bash
python scripts/unpack_shared_data.py --data-dir /your/writable/data --workers 4
bash scripts/train.sh --config configs/baseline.yaml --data-root /your/writable/data
```

The helper copies ordinary WAVs, pairings and reference metadata, then extracts
train-360. It checks archive size, extraction success and WAV counts/bytes;
completed packages are skipped on reruns when their counts and sizes match.
Existing incomplete training WAVs require `--overwrite`. Source data is left
intact when a different output directory is selected. Training checks the CSVs
and representative audio before allocating the model, and explains the unpack
step if train-360 is still packaged.

For another shared source location, pass `--source-data-dir /path/to/shared/data`
to the helper. For training and evaluation, `--data-root` overrides
`FLOWTSE_DATA_ROOT`, which overrides `data.root`. For example:

```bash
export FLOWTSE_DATA_ROOT=/your/writable/data
bash scripts/train.sh --config configs/baseline.yaml
```

You can also edit `data.root` or use an inherited YAML with
`extends: baseline.yaml`. Absolute paths outside the configured root remain
unchanged. All enrollment candidates, row order and candidate order are
preserved in the shared pairings; audio paths are relative to their CSV.
SHA-256 checks are disabled throughout this sharing version, including dataset
preparation, model loading and scoring. Historical checksum fields in existing
delivery manifests are informational and are not rechecked.

Mixture metadata uses the LibriMix columns `mixture_ID`, `mixture_path`,
`source_1_path`, `source_2_path`, and `length`. Lengths are samples at the input
sample rate, 16 kHz. Each enrollment CSV starts with a header and contains rows
in this form:

```csv
mixture_id,utterance_id,enrollment_path,enrollment_length
1-1-1_2-1-1,1-1-1,s1/1-1-2_3-1-1.wav,64000
```

`utterance_id` identifies the target source in `mixture_id`. Use a different
utterance from the same speaker for enrollment. Additional candidate path/length
pairs can follow the first pair on the same row. Relative audio paths are
resolved against the CSV's directory. To relocate absolute CSV paths, optionally
set `data.path_prefix_map`, for example `{"/old/datasets": "/new/datasets"}`.

### Dataset access without the shared directory

To prepare the audio yourself, follow the official
[LibriMix repository](https://github.com/JorisCos/LibriMix) and its
`generate_librimix.sh` instructions for downloading the source audio and
generating mixtures. Match this recipe: **Libri2Mix, two speakers, 16 kHz, min
mode, noisy `mix_both`**, with `s1`/`s2` references and the train-100, train-360,
DEV and TEST splits. Match the layout above or adjust the split and metadata
paths in your configuration to the generated layout.

The official repository supplies mixture-generation metadata, but does not
include this project's target-speaker enrollment mappings
(`mixture2enrollment.csv`). **We recommend using the enrollment pairing files
provided with this project**, even when generating the audio independently.
Place one in each split directory, retaining mixture/utterance IDs, candidate
order and matching audio paths. The supplied DEV/TEST ASR reference JSONs also
preserve the evaluation protocol used for the results above.

The unpack helper prepares the shared archive bundle. For self-generated audio,
configure your data root and split paths after adding the enrollment mappings.
Obtain the audio using the official LibriMix workflow above. If you cannot access
the shared enrollment mappings, **contact the repository maintainer for the
pairing files**, which can be uploaded to **Hugging Face on request**. The optional
Hugging Face distribution covers pairing files and the model checkpoint;
audio data should be downloaded/generated through the official repository.

### BigVGAN decoder weights

For the Mel baseline, obtain the compatible
[NVIDIA BigVGAN v2 weights](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
for `bigvgan_v2_24khz_100band_256x` and put `config.json` and
`bigvgan_generator.pt` in `pretrained/bigvgan_v2_24khz_100band_256x/`, or change
`vocoder.checkpoint_dir`. Loading checks the frontend configuration and model
parameter compatibility.

Run this manual preparation step in the training environment to download just
the two required decoder files:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
for filename in ('config.json', 'bigvgan_generator.pt'):
    hf_hub_download(
        repo_id='nvidia/bigvgan_v2_24khz_100band_256x',
        filename=filename,
        local_dir='pretrained/bigvgan_v2_24khz_100band_256x',
    )
PY
```

The decoder implementation is included in this repository. Installing a
separate BigVGAN repository or compiling its optional CUDA kernel is unnecessary.
Keep both decoder files available for training validation and later inference;
separator checkpoints do not embed the frozen BigVGAN weights.

### Checkpoint used for the reported TEST results

The checkpoint shared for inference and initialization is:

```text
/share/202611/FlowTSE-RL-share/checkpoints/baseline-epoch0460-step465060.ckpt
```

It contains the separator weights used for **all four NFE=4/8/16/32 TEST
evaluations** above. The checkpoint was selected by **DEV PESQ** from the
500-epoch baseline run: **460 completed epochs** (saved zero-based epoch 459),
**global step 465060**.

The shared export keeps the original FP32 model parameters and a portable
configuration. It is intended for inference or initializing subsequent
experiments; full SFT resume additionally requires optimizer, scheduler and RNG
states, which are excluded from this export along with the original W&B identity.
BigVGAN weights must still be prepared separately.

Pass the shared path to `--checkpoint` in the inference/evaluation commands
below. If you cannot access the shared directory, **contact the maintainer to
request a Hugging Face upload of this checkpoint**. Use the downloaded file's
local path with `--checkpoint`. Model weights are kept outside the GitHub code
repository.

## Training and resume

The supplied launcher sets CPU thread limits and starts ordinary PyTorch DDP
through torchrun. Accelerate handles mixed precision and DDP wrapping.

Prepare the complete audio dataset and BigVGAN files first. If you extracted
the data to another directory, set `FLOWTSE_DATA_ROOT` as shown above or pass
`--data-root` to each training command.

```bash
bash scripts/train.sh --config configs/baseline.yaml
NUM_GPUS=1 bash scripts/train.sh --config configs/baseline.yaml
```

Equivalent four-GPU launch:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/baseline.yaml
```

Single-GPU launch without torchrun:

```bash
python train.py --config configs/baseline.yaml --num-gpus 1
```

The full reference global batch is 128 (4 GPUs × 32 samples × accumulation 1);
tail batches can be smaller.
Changing the GPU count or per-GPU batch changes that global batch. To adapt
the recipe, put a YAML with `extends: baseline.yaml` next to the baseline and
override `data.max_samples_per_batch`, `train.accumulation_steps` or other fields
explicitly. Model compilation and worker defaults remain in `flowtse_rl/config.py`.

Outputs stay in `runs/baseline/`. Checkpoints are written
atomically to `checkpoints/`: rolling `last.ckpt` every epoch and the top three
validation PESQ checkpoints, with `best.ckpt` pointing to the best retained
candidate. Rerunning the same command automatically resumes `last.ckpt` when
`checkpoint.auto_resume` is enabled. Explicit resume:

```bash
bash scripts/train.sh --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.ckpt
```

Full resume restores model, optimizer, scheduler, progress and per-rank RNG state.
Keep the original world size, data protocol and training budget for an exact
resume. DataLoader worker prefetch queues are not saved, so a multi-worker
resume does not guarantee the same subsequent random crop sequence. Use a new
`--output-dir` for an independent experiment.

W&B support is retained but disabled by default (`train.wandb: false`). To
enable online logging, first log in with your own account:

```bash
wandb login
```

For a self-hosted W&B service, use `wandb login --host https://your-wandb-server`
instead. Your account must have access to the selected project or team.
Then set these fields in `configs/baseline.yaml` before launching training:

```yaml
train:
  wandb: true
  wandb_project: FlowTSE-RL
  wandb_name: baseline
```

With logging enabled, the run ID is saved beside the checkpoints and the chart
axis uses fractional epochs. To record offline instead, set `WANDB_MODE=offline`;
sync those runs later with the standard `wandb sync` command. Keep
`train.wandb: false` to train without W&B.

Acceleration defaults live in `flowtse_rl/config.py`: block compilation,
vectorized packing and the redundant-mask fast path are enabled, gradient
checkpointing is disabled, and the DataLoader uses 8 workers. Tail batches keep
an eager fallback and validation uses eager inference. These execution options
can be overridden explicitly when adapting to another machine.

## Inference and evaluation

```bash
python infer.py --checkpoint runs/baseline/checkpoints/best.ckpt \
  --mixture mixture.wav --enrollment enrollment.wav --output extracted.wav --steps 4

python evaluate.py --checkpoint runs/baseline/checkpoints/best.ckpt \
  --config configs/baseline.yaml \
  --output-dir results/baseline-nfe4 --split test --steps 4 --save-audio

bash scripts/evaluate_nfe.sh runs/baseline/checkpoints/best.ckpt results/nfe \
  --config configs/baseline.yaml --split test
```

The NFE script evaluates 4, 8, 16 and 32 Euler steps sequentially with the same
checkpoint and per-item seeds. Inference is FP32 by default. Use `--device cpu`
for small checks. A checkpoint stores its model/frontend configuration. Passing
`--config` allows relocating data and vocoder paths while checking architecture
and frontend compatibility.

These commands require a trained separator checkpoint. `infer.py` accepts an
arbitrary mixture WAV and enrollment WAV; it resamples to the configured sample
rate, uses up to the first 3 seconds of the supplied enrollment and writes a
mono 24 kHz output for the baseline. `evaluate.py` uses the
prepared DEV/TEST pairings. It evaluates the entire selected split by default
(6000 target pairs in each shared split), independently of training's 20% DEV
subset. For a quick check, use a new output directory and add `--limit 8`.
Standalone evaluation runs in one process; the NFE script runs the four settings
sequentially. Append `--save-audio` to that script when retaining its audio.

Inference computes the complete mixture Mel once, splits the features into
3-second Flow windows, joins the predicted features, then decodes the entire
utterance once with BigVGAN. The full output length is preserved and all
metrics are computed on complete waveforms.

Built-in evaluation measures SI-SDR, wideband PESQ and ESTOI. It writes per-item
CSV results, metric coverage, checkpoint path/size and `summary.json`; successful
completion adds `COMPLETE.json`. `--limit` selects an explicitly reported subset.
Use `--save-audio` to retain waveforms. Output directories must be new.

With `--save-audio`, the example above keeps matching files under
`results/baseline-nfe4/audio/`:

```text
<mixture_id>__target-<utterance_id>_pred.wav
<mixture_id>__target-<utterance_id>_ref.wav
```

The standalone scorer expects this flat paired layout, with identical sample
rates and lengths within each pair. Without `--save-audio`, evaluation removes
the temporary pairs after scoring, so they cannot be reused for another scorer.

The bundled scorer also measures DNSMOS OVRL/SIG/BAK/P.808, true-text WER/CER,
WavLM cosine similarity and WeSpeaker ResNet221-LM similarity: 11 metrics in
all. Score already generated audio independently of inference:

```bash
python score.py --audio-dir results/baseline-nfe4/audio --output-dir results/scores --metrics core --workers 4
python score.py --audio-dir results/baseline-nfe4/audio --output-dir results/full-scores --metrics full \
  --scoring-device cuda --dnsmos-dir pretrained/dnsmos \
  --qwen-model pretrained/qwen3-asr --wer-reference transcripts \
  --transcripts-json "${FLOWTSE_DATA_ROOT:-/share/202611/FlowTSE-RL-share/data}/asr_references/test.json" \
  --wavlm-model pretrained/wavlm-base-plus-sv \
  --wespeaker-model pretrained/wespeaker/voxceleb_resnet221_LM.onnx
```

Use the `flowtse-score` environment from the [installation steps](#installation).
Its pinned scoring stack includes PyTorch 2.8.0, Transformers 4.57.6,
Qwen-ASR 0.0.6 and whisper-normalizer 0.1.15. Model files remain separate from
the installed Python packages.

Required local assets are:

| Metric | Assets |
| --- | --- |
| DNSMOS and P.808 | `sig_bak_ovr.onnx` and `model_v8.onnx` from [DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS) |
| WER/CER | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model/processor directory and the supplied ASR reference JSON, or original LibriSpeech audio/transcripts |
| WavLM similarity | [wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) model/processor directory |
| WeSpeaker similarity | [wespeaker-voxceleb-resnet221-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM), revision `26732f43821f238bfb258c5ef54924a798e62118`, file `voxceleb_resnet221_LM.onnx` |

The example flags use this local layout:

```text
pretrained/
  bigvgan_v2_24khz_100band_256x/{config.json,bigvgan_generator.pt}
  dnsmos/{sig_bak_ovr.onnx,model_v8.onnx}
  qwen3-asr/                 # Complete model, tokenizer and processor files
  wavlm-base-plus-sv/        # Complete model and feature-extractor files
  wespeaker/voxceleb_resnet221_LM.onnx
```

Download the complete Qwen3-ASR and WavLM model directories from the linked
repositories, including their configuration and preprocessing files. Only the
two listed ONNX files are needed for DNSMOS, and only the listed ONNX export
for WeSpeaker. These assets are optional for core SI-SDR/PESQ/ESTOI scoring.

All scoring model loads are local; there are no automatic downloads. The environment
variables `DNSMOS_DIR`, `QWEN_ASR_MODEL_ID`, `WAVLM_MODEL_PATH`,
`WESPEAKER_MODEL_PATH` and `LIBRISPEECH_ROOT` can supply the corresponding paths.
Run the same options with `--check-assets` to check setup without scoring.
Metric groups `core`, `dnsmos`, `asr`, `speakers` and `full` can be combined,
for example `--metrics core,dnsmos`. Optional models load only when requested.
DNSMOS, ASR, WavLM and WeSpeaker execute sequentially; DNSMOS defaults to one
worker. CPU PESQ/ESTOI use independent processes controlled by `--workers`.

The bundled scorer uses LibriMix's complete-shorter-target true-text
ASR protocol. Longer or cropped targets retain all quality scores but are
excluded from WER/CER, with explicit skip reasons. WER/CER are fractions,
rounded per utterance to three decimals and then macro-averaged. The
`whisper-normalizer` English text normalization is fixed. VCTK true-text
references are supported via `--wer-reference vctk-shorter --vctk-root ...`,
with original durations as needed. For another dataset, choose
`--wer-reference transcripts --transcripts-json references.json`; map each
`*_ref.wav` filename to its true text, or to an object containing
`reference_text`, `wer_eligible` (a JSON boolean), and `wer_skip_reason`.

`_results.csv`, `protocol.json` and `summary.json` preserve per-item values,
model metadata and each metric's eligible/finite/failed/skipped counts. Missing
models stop scoring; they are not replaced with placeholder scores. Audio
length mismatches are rejected. `COMPLETE.json` is written only after every
requested stage finishes. Use a new output directory for another scoring run.

To select this bundled scorer after inference, add `--scoring-metrics` and the
same asset options to `evaluate.py`:

```bash
export FLOWTSE_SCORING_PYTHON="$(conda run -n flowtse-score python -c 'import sys; print(sys.executable)')"
python evaluate.py --checkpoint runs/baseline/checkpoints/best.ckpt \
  --config configs/baseline.yaml --split test \
  --output-dir results/baseline-full-nfe4 --steps 4 --scoring-metrics full --scoring-device cuda \
  --dnsmos-dir pretrained/dnsmos --qwen-model pretrained/qwen3-asr \
  --wavlm-model pretrained/wavlm-base-plus-sv \
  --wespeaker-model pretrained/wespeaker/voxceleb_resnet221_LM.onnx
```

With `--scoring-metrics`, `evaluate.py` automatically uses
`data.asr_reference_dir/{dev,test}.json` when no explicit transcript JSON or
LibriSpeech root is supplied. These mappings preserve true-text eligibility:
3000/6000 DEV and 3004/6000 TEST targets. `score.py` accepts the same mappings
with `--wer-reference transcripts --transcripts-json ...`, as shown above.
The LibriSpeech notice is included in the shared `data/licenses/` directory.

For that command, supply the asset paths through the flags above or environment
variables. The inference and scoring processes can use different Python
environments. For training-time scoring, set `validation.run_external_metrics`
to `true` and add `validation.scoring_options` using the fields in
`flowtse_rl/scoring/runner.py:Options`, for example `metrics: [core, dnsmos]`,
`device: cuda`, and `dnsmos_dir: pretrained/dnsmos`. No separate backend code is
required. `validation.metrics_command` can launch a custom scorer and takes
precedence when explicitly configured.

For ASR during training validation, also configure `wer_reference: transcripts`,
`transcripts_json` pointing to the prepared data root's `asr_references/dev.json`,
and the local Qwen model path. Automatic DEV/TEST JSON selection described above
applies to `evaluate.py`; training-time scoring uses the supplied options.

Without `--scoring-metrics` or `validation.scoring_options`, evaluation uses
the compact core scorer. The bundled scorer records its own protocol ID; its
Librosa resampling/peak normalization can differ numerically from the compact
Torchaudio path. Compare runs using the same scorer/protocol.

## License and attribution

The repository uses the [MIT license](LICENSE), Copyright (c) 2025 Tsun-An Hsieh,
except for components carrying their own license notices:

- The UDiT source retains the original Meta Platforms copyright notices and
  upstream references in its file headers.
- [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN) retains its
  [MIT license](flowtse_rl/vendor/bigvgan/LICENSE) and bundled dependency licenses
  under `flowtse_rl/vendor/bigvgan/incl_licenses/`. Imports are package-qualified;
  model weights are not redistributed.
- Worker seeding adapts the SeedSequence protocol from
  [Lightning Fabric 2.0.6](https://github.com/Lightning-AI/pytorch-lightning/tree/2.0.6),
  by the Lightning AI authors, for this trainer. See
  [Apache License 2.0](licenses/Lightning-Apache-2.0.txt).
- DNSMOS/P.808 scoring adapts [SoloSpeech](https://github.com/WangHelin1997/SoloSpeech),
  Copyright (c) 2025 Helin Wang, under
  [CC BY-NC 4.0](licenses/SoloSpeech-CC-BY-NC-4.0.txt). This component retains that
  license. Adaptations isolate optional models, select CPU/CUDA providers,
  check full input lengths and record metric coverage. Calibration follows
  [Microsoft DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

Data and pretrained weights remain subject to their respective licenses.
