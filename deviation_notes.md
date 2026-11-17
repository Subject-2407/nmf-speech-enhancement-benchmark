# Implementation Deviations & Technical Notes

### 1. Library API Changes (`scikit-learn >= 1.2`)

In the original `main.py`, the batch `NMF` baseline was instantiated using `NMF(..., alpha=0.0)`. However, in `scikit-learn 1.8.0` (and versions $\ge 1.2$), the `alpha` argument was removed, raising `TypeError: NMF.__init__() got an unexpected keyword argument 'alpha'`.

**Resolution:**
An adaptive `try-except` fallback using `alpha_W=0.0` is implemented in `main.py` and the replication notebook to ensure compatibility across both legacy and modern `scikit-learn` versions.

---

### 2. Global Variable Bug in the Original Repository (`main.py`)

In the upstream `Denoising(training_data, test_data, ...)` function in `main.py`, the NMF branch looped over a global variable `training` instead of the local argument `training_data`. Consequently, passing a custom list of training files (e.g., switching between White Noise and Air Noise) failed to update the training noise source.

**Resolution:**
The loop in `main.py` and `01_baseline_replication.ipynb` now iterates directly over `training_data`, enabling dynamic switching across noise conditions (`WhiteNoise.wav` and `AirNoise.wav`).

---

### 3. Evaluation Pairing Inconsistency in `performance.py`

In the `__main__` demo block of upstream `performance.py`, the script paired `voice_nmf` with `noise_onmf` across different algorithms.

**Resolution:**
In all evaluation notebooks, separated speech and noise pairs are strictly matched within the same method (`voice_nmf` with `noise_nmf`, and `voice_onmf` with `noise_onmf`).

---

### 4. Signal Dimension Alignment

Original clean/noisy recordings contain 935,526 samples (truncated to match the shorter mixture when needed), whereas iSTFT reconstruction produces 935,936 samples due to STFT hop-size frame padding.

**Resolution:**
The evaluation pipeline automatically truncates synthesized and reference arrays to `min(len(synth), len(orig))` prior to orthogonal QR projection in BSS Eval (`SDR`, `SIR`, `SAR`) and resampling for `PESQ` / `STOI`.

---

### 5. Numerical Stability & Deterministic Random Seed

The upstream `update_code_within_radius` function in `onmf.py` divides by `np.linalg.norm(H1_old, 2)`, which can emit `RuntimeWarning: invalid value encountered in scalar divide` when the code norm approaches zero. Furthermore, the original code did not fix a random seed (`seed=None`), introducing run-to-run patch-sampling variance.

**Resolution:**
- Added a numerical stabilization term $\varepsilon = 10^{-12}$ in the denominator of the stopping criterion.
- Fixed `RANDOM_SEED = 42` globally in NumPy and `scikit-learn` (`random_state=42`) to ensure full reproducibility, yielding low replication variance relative to the published baseline tables:
  - White Noise SDR/SAR deviation: $< 0.35\text{ dB}$.
  - Air Noise metric deviation: $< 0.05\text{ dB}$.
