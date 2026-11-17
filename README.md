# NMF Speech Enhancement Benchmark

An apples-to-apples comparative benchmark of **supervised Non-Negative Matrix Factorization (NMF)** methods for **single-channel speech enhancement** (`fs = 48,000 Hz`) under stationary broadband Gaussian White Noise and non-stationary real-world Air Purifier noise (*Levoit-H132*).

> **Associated Paper:** *Link forthcoming (to be updated upon publication).*

![Supervised NMF Architecture Flowchart](benchmarks/figures/architecture_flowchart.png)

---

## Benchmarked Methods

All three methods operate strictly within the **supervised prior-dictionary paradigm** trained on clean speech (`Data/audio/whitman.wav`) and isolated noise (`WhiteNoise.wav` / `AirNoise.wav`), evaluated on `shakespeare.wav` mixtures:

1. **Baseline — `ONMF-Frobenius`** *(Sack et al., IEEE CISS 2022)*: Online NMF with Euclidean (Frobenius) objective, patch streaming, and $\ell_1$-sparse projected gradient descent ($\alpha = 100$).
2. **Comparator 1 — `ONMF-KL`** *(Audited EM/MM Extension of Sack et al., Sec. 4)*: Scale-invariant Generalized Kullback-Leibler Divergence ONMF using exact online EM/MM sufficient statistics ($A_t, B_t$) and tunable speech activation sparsity ($\alpha_s$).
3. **Comparator 2 — `NMF-HMM-KL`** *(Xiang et al., EURASIP JASMP 2022)*: Dynamic multi-state Speech & Noise HMMs under Poisson/KL likelihoods, Kronecker state composition ($\bar{\boldsymbol{\pi}} \otimes \ddot{\boldsymbol{\pi}}, \bar{\mathbf{A}} \otimes \ddot{\mathbf{A}}$), causal forward posteriors, and Eq. (46) MMSE spectral gain.

---

## Dual-Track Evaluation Summary (`5-Fold Cross-Segmentation`, `fs = 48 kHz`)

| Noise Class | Method | Default Track (`2048/1024`)<br/>SDR (dB) / PESQ / STOI | Joint Tuned (`4096/3072`)<br/>SDR (dB) / PESQ / STOI | Inference<br/>(s / fold) |
| :--- | :--- | :---: | :---: | :---: |
| **White Noise** | `Noisy Input` | `9.51` / `1.339` / `0.909` | — | — |
| | `Baseline (ONMF-Frobenius)` *(Def / Ctrl)* | `22.72` / `2.526` / `0.887` | `20.90` / `2.156` / `0.866` | `2.72` / `3.07` |
| | `Comparator 1 (ONMF-KL)` | **`24.47`** / `3.010` / **`0.927`** | **`24.01`** / **`3.660`** / `0.928` | **`0.16`** / **`0.52`** |
| | `Comparator 2 (NMF-HMM-KL)` | `24.20` / **`3.036`** / **`0.927`** | `23.97` / `3.207` / **`0.933`** | `0.40` / `5.99` |
| **Air Noise** | `Noisy Input` | `5.65` / `1.407` / `0.710` | — | — |
| | `Baseline (ONMF-Frobenius)` *(Def / Ctrl)* | **`10.17`** / `1.777` / `0.746` | `11.05` / `1.832` / `0.758` | `3.76` / `4.56` |
| | `Comparator 1 (ONMF-KL)` | `9.28` / `1.682` / `0.743` | **`11.42`** / `2.012` / `0.781` | **`0.18`** / **`0.52`** |
| | `Comparator 2 (NMF-HMM-KL)` | `10.05` / **`1.858`** / **`0.756`** | `11.20` / **`2.144`** / **`0.784`** | `0.38` / `4.34` |

Key takeaways:
- **Joint Cross-Noise Superiority (`4096/3072`, 75% Overlap):** Under the Joint Tuned STFT grid (`nperseg=4096, noverlap=3072`, matching the `21.3 ms` temporal hop of Default `2048/1024` while doubling frequency resolution to `11.7 Hz/bin`), **both comparators (`ONMF-KL` with `K=[100,15], alpha_s=0.08, q=1.3` and `NMF-HMM-KL` with `arch=(8,25,4,20), alpha_s=0.05`) significantly outperform both Standard Baseline (`2048/1024`) and Baseline Control (`4096/3072`) across all 6 metrics** (`SDR`, `PESQ`, `STOI` on both White Noise and Air Noise; **24/24 Holm-adjusted $t$-tests significant**, $p_{\text{Holm}} \le 0.0287$).
- **`ONMF-KL` (`SDR` + Speed + Low False-Positive Leader):** Eliminates `ONMF-Frobenius` latency (`0.16–0.52 s/fold` vs. `2.72–4.56 s/fold`), leads Tuned White Noise PESQ (`3.660`), Tuned White Noise SDR (`24.01 dB`), Tuned Air Noise SDR (`11.42 dB`), and Air Noise silence leakage reduction (`VAD F1 = 0.952`, `FP = 25` vs. `31` Control and `241` untuned KL).
- **`NMF-HMM-KL` (Perceptual & Intelligibility Leader in Real-World Air Noise):** Tracks sequential speech/noise state transitions to achieve the **highest perceptual PESQ on non-stationary Air Noise** in both Default (`1.858`) and Tuned (`2.144`) tracks as well as top Tuned STOI (`0.933` White / `0.784` Air).

---

## Repository Structure

```text
.
├── 01_baseline_replication.ipynb         # End-to-end replication of Sack et al. (2022)
├── benchmarks/
│   ├── 02_comparative_experiments.ipynb  # Dual-track full-clip runs, spectrograms & audio demos
│   ├── 03_statistical_analysis.ipynb     # 5-fold evaluation, Friedman, Holm t-Test, Pearson r, VAD & validation curves
│   ├── generate_architecture_diagram.py  # Reproducible matplotlib architecture flowchart generator
│   ├── figures/                          # Publication figures & diagnostic plots
│   └── audio_output/                     # Separated .wav outputs for Baseline, ONMF-KL, and NMF-HMM-KL
├── Data/
│   ├── audio/                            # Clean speech, noise priors, and noisy mixtures (fs = 48 kHz)
│   └── audio_output/                     # Precomputed & replicated baseline waveforms
├── onmf.py                               # Core Online NMF module (Lyu et al., 2020)
├── dictionary_learner.py                 # Patch-based ONMF dictionary learning wrapper
├── audio_enhancement.py                  # Sparse-coding separation and Wiener mask synthesis
├── main.py                               # Baseline CLI script (with scikit-learn>=1.2 compatibility fix)
├── performance.py                        # BSS Eval (SDR, SIR, SAR) calculation module
├── deviation_notes.md                    # Audit log of upstream bugs, API fixes & seed determinism
├── experiment_log.txt                    # Hardware, runtime dependencies, and hyperparameter configurations
├── requirements.txt                      # Python package requirements
├── README_BASELINE.md                    # Original upstream repository README
└── LICENSE                               # MIT License
```

---

## Quick Start

1. **Install dependencies** (tested on Python 3.13.0):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run notebooks in order**:
   - [`01_baseline_replication.ipynb`](01_baseline_replication.ipynb) — Reproduces the `NMF` and `ONMF` baseline tables and spectrograms.
   - [`benchmarks/02_comparative_experiments.ipynb`](benchmarks/02_comparative_experiments.ipynb) — Runs Default (`2048/1024`) and Joint Tuned (`4096/3072`) comparisons across `ONMF-Frobenius`, `ONMF-KL`, and `NMF-HMM-KL`.
   - [`benchmarks/03_statistical_analysis.ipynb`](benchmarks/03_statistical_analysis.ipynb) — Generates 5-fold statistical significance tests, correlation analyses, VAD confusion matrices, boxplots, and validation curves.
3. **Regenerate the architecture diagram** *(optional)*:
   ```bash
   python benchmarks/generate_architecture_diagram.py
   ```

---

## References & Acknowledgments

- **Baseline Paper & Upstream Code**: Andrew Sack, Wenzhao Jiang, Michael Perlmutter, Palina Salanevich, and Deanna Needell, *"On Audio Enhancement via Online Non-negative Matrix Factorization"*, IEEE CISS 2022 ([arXiv:2110.03114](https://arxiv.org/abs/2110.03114)); upstream implementation `Jerry-jwz/Audio-Enhancement-via-ONMF` (preserved in [`README_BASELINE.md`](README_BASELINE.md)).
- **Core ONMF Algorithm**: Hanbaek Lyu, Georg Menz, Deanna Needell, and Christopher Strohmeier, *"Applications of online nonnegative matrix factorization to image and time-series data"*, IEEE ITA Workshop, 2020 ([IEEE Xplore](https://ieeexplore.ieee.org/document/9245004), [GitHub](https://github.com/HanbaekLyu/ONMF_ONTF_NDL.git)).
- **Temporal HMM-NMF Comparator (`NMF-HMM-KL`)**: Xiang et al., *EURASIP Journal on Audio, Speech, and Music Processing* (2022).
- **License**: Released under the MIT License ([`LICENSE`](LICENSE)).
