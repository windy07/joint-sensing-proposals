# Learning to Propose Joint Sensing Actions under Shared Uncertainty

Code, pretrained checkpoints, and compact experimental results for joint radar sensing.
The model proposes sensing controls and ranks candidate action groups. A shared
physical forecast evaluates the candidates and selects the executable group.

This release contains the scientific implementation used in the completed
experiments, three trained encoder variants with three initialization seeds each,
and episode-level results for the development and confirmation evaluations.

## Contents

| Material | Location |
|---|---|
| Physical model, tracking, action construction, scheduling, and neural models | [`project/ACTIVE/MAINLINE_RESEARCH/spatial_scheduling/spatial_lab`](project/ACTIVE/MAINLINE_RESEARCH/spatial_scheduling/spatial_lab) |
| Training configuration, scenario definitions, and world splits | [`project/STAGE6_SERVER/plan_12800.json`](project/STAGE6_SERVER/plan_12800.json) |
| Training and data-generation controller | [`project/stage6_server.py`](project/stage6_server.py) |
| Training adapters used for the ten-epoch run | [`run/EPOCH10_RUN/ADAPTER_SOURCE`](run/EPOCH10_RUN/ADAPTER_SOURCE), [`linux_adapter`](linux_adapter) |
| Evaluation methods, inference-time ablations, and paired statistics | [`posttrain_confirmation`](posttrain_confirmation) |
| Selected checkpoints and download identities | [`checkpoints/manifest.json`](checkpoints/manifest.json) |
| Confirmation episode summaries and aggregate results | [`results/confirmation`](results/confirmation) |
| Development episode summaries | [`results/development`](results/development) |
| Training loss histories and epoch-ten summaries | [`results/training`](results/training) |

Full world trajectories, prepared training tensors, optimizer snapshots, and
server operation logs are excluded. The scenario generator and fixed world seeds
are included. The released episode summaries are sufficient to recompute the
reported paired statistics without rerunning the simulator.

## Environment

Use Python 3.12. The original environment used PyTorch 2.10.0, NumPy 2.3.5,
SciPy 1.17.0, and threadpoolctl 3.6.0. The checkpoint loader checks these public
package versions; CPU and CUDA build suffixes are recorded separately.

For a Linux CUDA 12.8 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/verify_release.py --environment
```

For CPU evaluation, install the CPU build of the same PyTorch version using
`https://download.pytorch.org/whl/cpu` as the wheel index. Simulation evaluations
use Linux. Source verification and inspection of saved results also work on
Windows. GPU choice and concurrency depend on the available machine.

## Pretrained checkpoints

Download the nine assets from [release v0.1.0](https://github.com/windy07/joint-sensing-proposals/releases/tag/v0.1.0):

```bash
python scripts/download_checkpoints.py
python scripts/verify_release.py --models-root run/models --environment
```

For a private repository, the downloader uses `GH_TOKEN`, `GITHUB_TOKEN`, or the
existing GitHub CLI login. Alternatively, download the release assets in the
browser and install them locally:

```bash
python scripts/download_checkpoints.py --from-dir checkpoint_assets
```

The assets are installed as `run/models/<arm>/<seed>/best.pt`. The downloader
checks the file size and SHA-256 value and does not replace a different existing
checkpoint. The nine files total about 467 MiB. They contain model weights and
the metadata needed by the original loader, without optimizer state.

| Encoder | Code name | Seed 633431 | Seed 633432 | Seed 633433 |
|---|---|---:|---:|---:|
| Relational | `integrated_agent` | epoch 10 | epoch 10 | epoch 10 |
| Deep Sets | `deep_sets` | epoch 10 | epoch 10 | epoch 10 |
| Self-attention | `set_transformer` | epoch 9 | epoch 10 | epoch 9 |

Each checkpoint was selected by minimum validation loss among completed epochs
1–10, with the earliest epoch breaking a tie. These are the checkpoints used in
the final confirmation evaluation. `set_transformer` is the implementation name
for the Self-attention comparison in the paper.

## Evaluate a released model

The portable entry point selects a job from the original confirmation design.
First inspect the selected job; this performs no simulation:

```bash
python scripts/evaluate.py --arm integrated_agent --model-seed 633431 --world-seed 623000001 --variant n1f1
```

On Linux, add `--execute` to run that episode:

```bash
python scripts/evaluate.py --arm integrated_agent --model-seed 633431 --world-seed 623000001 --variant n1f1 --device cuda:0 --execute
```

Run either baseline on the same world:

```bash
python scripts/evaluate.py --world-seed 623000001 --variant numeric_optimizer --execute
python scripts/evaluate.py --world-seed 623000001 --variant fixed_analytic --execute
```

The entry point uses the original model loader, action policy, physical forecast,
world generator, and frozen job configurations. It verifies the selected weight
against the confirmation manifest. It uses the released best checkpoint directly;
the original archived queue additionally expects the epoch-ten audit checkpoint.

Each execution writes one receipt under `outputs/confirmation/receipts`.
`--save-trace` additionally saves the full episode trace. `--output` selects a
different destination. `--job <id>` selects any exact job from the frozen design.
An existing receipt is preserved; use a fresh output directory for a rerun.

| Variant | Numerical controls | Within-cycle feedback |
|---|---|---|
| `n0f0` | Preset | Off |
| `n1f0` | Learned | Off |
| `n0f1` | Preset | Candidate reranking |
| `n1f1` | Learned | Candidate and control refinement |

The full four-way comparison is available for `integrated_agent`. The other
encoders have the `n0f0` and `n1f1` jobs included in the frozen evaluation design.

## Reproduce result summaries and tables

The development set contains 64 worlds and 1,664 evaluated episodes. The
confirmation set contains 192 worlds and 4,992 evaluated episodes. Each set
covers eight scenarios; an episode is one method and model seed evaluated in
one world. The scenario, world seed, method, and model seed are retained in
`episode_metrics.csv` and `episode_summaries.json.gz`.

Recompute statistics from the saved confirmation summaries:

```bash
python scripts/summarize.py --output outputs/statistics.json
python posttrain_confirmation/paper_tables.py --report outputs/statistics.json --output outputs/tables
```

To export tables immediately from the archived statistics:

```bash
python posttrain_confirmation/paper_tables.py --report results/confirmation/statistics.json --output outputs/tables
```

The exporter produces analysis tables; their typography may differ from the
typeset paper. To summarize newly generated complete evaluation receipts, pass
`--receipts outputs/confirmation/receipts` to `scripts/summarize.py`.

The statistics first average the trained seeds within each world and method,
then weight the eight scenarios equally. Resampling is paired by world within
scenario. Definitions and units are stored with every metric in
`results/confirmation/statistics.json`.

## Training configuration and source organization

The training set contains 12,800 worlds; the validation set contains 128 separate
worlds. Every world has 64 cycles. The batch size is 32 worlds and the temporal
gradient window is eight cycles, giving 3,200 optimizer updates per epoch.
Each of the nine fits ran through epoch 10, or 32,000 updates, while retaining
the original 20-epoch learning-rate schedule.

The model has hidden and memory width 384, six encoder layers, and eight attention
heads for the attention-based encoders. The original configuration is in
`plan_12800.json`; the ten-epoch stopping logic is in `epoch10_training.py`.
The Linux and input adapters document the actual training execution. Their
hardware profiles describe the original machines and need a new execution
configuration on a different host. Prepared training tensors are generated
separately and are not distributed here.

The nested source layout and line endings are retained because the original
checkpoints bind to those source files. `scripts/` contains the portable release
entry points; it is separate from that preserved source. Run
`python scripts/verify_release.py` to check the copied source and configuration
files against `provenance/original_files.json`.

## Publication status

This is the authors' repository for maintaining the release. An anonymous review
copy must use a separate anonymous entry point, including anonymous download links.
No open-source license has been assigned in this repository yet.
