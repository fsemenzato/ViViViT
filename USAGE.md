# ViViViT: usage

Install with `python -m pip install -e .`. Python 3.10+ and PyTorch 2.1+ are required. Simulation generation additionally needs py21cmfast 3.x, which is not installed by this package.

## Main experiment

Run commands from this folder. Training configs use relative paths; set `preproc_video_dir` and `output_base` to your data and output locations.

```sh
python -m vivivit.generate --config configs/simulation.json --output data/raw --cache data/cache --ids 0 3000 --seed 42
python -m vivivit.preprocess --root data/raw --output data/train

torchrun --standalone --nproc_per_node=4 -m vivivit.train --config configs/vivit.json
torchrun --standalone --nproc_per_node=4 -m vivivit.train --config configs/unet.json

python -m vivivit.infer --checkpoint checkpoints/vivit/best_model.pth --data data/evaluation --output runs/recon/vivit --ids 0 200
python -m vivivit.infer --checkpoint checkpoints/unet/best_model.pth --data data/evaluation --output runs/recon/unet --ids 0 200

python -m vivivit.estimate power --source runs/recon/vivit --output runs/power --tag vivit_MLP --taper 0.5
python -m vivivit.estimate power --source runs/recon/unet --output runs/power --tag unet --taper 0.5
python -m vivivit.plot_power --data runs/power --output runs/figures/power.pdf

python -m vivivit.estimate bispectrum --source runs/recon/vivit --output runs/bispectrum/vivit
python -m vivivit.estimate bispectrum --source runs/recon/unet --output runs/bispectrum/unet
python -m vivivit.plot_bispectrum --data runs/bispectrum --output runs/figures/bispectrum.pdf
python -m vivivit.plot_bispectrum --data runs/bispectrum --output runs/figures/bispectrum_longmode.pdf --long-mode

python -m vivivit.plot_fields --vivit runs/recon/vivit/recon_sim_0000.pt --unet runs/recon/unet/recon_sim_0000.pt --output runs/figures/fields.pdf
```

Generate a separate evaluation pool with a different seed, then preprocess it to `data/evaluation`. ID ranges exclude the upper endpoint. The example generates a new 3,000-simulation pool; it does not regenerate the exact historical training set, whose surviving log reports 3,156 files. Generation writes per-simulation seeds, astrophysical parameters and simulator version into HDF5 attributes. `--format npy` supports the historical raw NumPy format during preprocessing. Files are reversed along the lightcone axis, cut at frame 300 to depth 384, and spatially resized to 128². Existing raw/preprocessed simulations cause an error instead of being overwritten.

The main trainer uses L1 pixel + L1 gradient loss, FP16 CUDA autocast, AdamW with betas (0.9, 0.95), batch 1 per GPU and accumulation 4. It writes `split.json`, `config.json`, `best_model.pth`, and a full resume checkpoint every five epochs. Resume with `--resume runs/vivit/resume_checkpoint.pth`. Inference uses the evaluation directory's `global_stats.json`, preserving the historical protocol.

Power spectra use three 128³ slabs of 512 Mpc. With `--taper 0.5` (the paper's estimator, Appendix on summary statistics) each slab is mean-subtracted, the window-weighted mean is removed, the slab is multiplied by a Tukey window with taper fraction 0.5 along the line of sight, no mass-assignment deconvolution is applied, modes are binned in shells of width k_F (b k_F ≤ |k| < (b+1) k_F) with independent-mode weights, and spectra are divided by mean(w²) of the 128-frame window. Omitting `--taper` selects the legacy estimator of earlier drafts (CIC compensation, rounded shells, no taper). `settings_<tag>.json` records which estimator wrote each product. The bispectrum uses the same slabs, five orientation bins and 42 shells; residual field and slab means are subtracted. The paper's bispectrum figures use the legacy setting (CIC, no taper); `--taper 0.5` applies the same window with mean(w³) normalization and no CIC. Bispectrum estimation is intended for a sufficiently large GPU.

`plot_power` and `plot_fields` reproduce the paper figures in their published style (LaTeX fonts when `latex` is on the PATH); the bispectrum plots keep the numerical selections with a simplified layout. Power axes convert physical Mpc⁻¹ to h/Mpc using h=0.6727, and only modes up to the Nyquist frequency are drawn. Columns run from the last slab (lowest redshift) to the first. Redshift labels come from `configs/lightcone_redshifts.txt`, the redshift of each of the 384 stored frames (frame 0 is z = 16.00). The paper's field-slice figure is the central transverse slice of held-out evaluation simulation 0.

## Loss ablation

Only the two ViViT arms shown in the paper are included. They use a separate reduced transformer with learned queries and SDPA attention. Set the configs' data path to the original ablation pool if recovered. The shared loader takes its center 128³ crop.

```sh
python experiments/loss_ablation/train.py --config experiments/loss_ablation/configs/L1.json
python experiments/loss_ablation/train.py --config experiments/loss_ablation/configs/L2.json
python experiments/loss_ablation/infer.py --checkpoint /path/to/best_model.pth --data /path/to/ablation-evaluation --output runs/loss_ablation/recon/L1
python experiments/loss_ablation/estimate.py --source runs/loss_ablation/recon/L1 --output runs/loss_ablation/power --tag ablation_vivit_128cube_L1_1gpu
python experiments/loss_ablation/plot.py --data runs/loss_ablation/power --output runs/figures/loss.pdf
```

Repeat inference/estimation for L2. This estimator uses the historical Hann window along the lightcone axis and divides spectra by mean(window²). Inference selects IDs 1086–1112. Both norms apply to the pixel and gradient terms. Unlike the main experiment, the historical ablation spectrum arrays have shape (simulation, k), with no slab dimension.

## Temporal memory

Only the four published architectures at M95=32,64,128,256 are retained: ViViT with causal attention and U-Nets with time-local GroupNorm at depths 2,3,4. Their saved configs are in `experiments/temporal_memory/configs`. Causal attention does not make the complete ViViT strictly causal: the convolutional decoder and its normalization also mix frames.

```sh
python experiments/temporal_memory/generate.py --output data/temporal/M32 --M95 32 --ids 0 500 --seed 0 --mode train
python experiments/temporal_memory/generate.py --output data/temporal/M32 --M95 32 --ids 500 700 --seed 1 --mode eval
python experiments/temporal_memory/train.py --config experiments/temporal_memory/configs/cml1_vivit_causal_M32.json
python experiments/temporal_memory/infer.py --checkpoint /path/to/best_model.pth --data data/temporal/M32 --output runs/temporal/cml1_vivit_causal_M32/recon
python experiments/temporal_memory/analyze.py --runs runs/temporal --output runs/temporal/analysis.npz
python experiments/temporal_memory/plot.py --analysis runs/temporal/analysis.npz --output runs/figures/temporal.pdf
```

Repeat for all 16 supplied configs. The generation counts/seeds above are a new reproducible example, not a claim about the missing original dataset manifests. Evaluation simulations are excluded using `manifest.json`. The trainer retains BF16 autocast, L1+GDL model selection, the original single-device AdamW defaults, accumulation, and optimizer-step cosine schedule. Analysis uses the ratio of summed squared errors to summed target power per frame; the sweep averages frames 96–159. `plot_slices.py` accepts the original saved sample filenames and renders simulation 500 at M95=256. No unused TCN, bidirectional, identity-control, hybrid, reverse-task or past/future experiment is included.

## Checks

```sh
python tests/test_release.py
```
