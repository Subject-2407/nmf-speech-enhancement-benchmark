import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_architecture_diagram(output_path=None):
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(base_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        output_path = os.path.join(fig_dir, "architecture_flowchart.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = plt.figure(figsize=(16, 11.2), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.2)
    ax.axis('off')

    c_bg = "#F8FAFC"
    c_title = "#0F172A"
    c_blue_border, c_blue_fill = "#2563EB", "#EFF6FF"
    c_amber_border, c_amber_fill = "#D97706", "#FFFBEB"
    c_green_border, c_green_fill = "#059669", "#ECFDF5"
    c_purple_border, c_purple_fill = "#7C3AED", "#F5F3FF"
    c_gray_border, c_gray_fill = "#64748B", "#F1F5F9"

    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    ax.text(8.0, 10.68, "SUPERVISED NMF ARCHITECTURE & DUAL-TRACK BENCHMARK PIPELINE",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=c_title)
    ax.text(8.0, 10.30,
            "Non-Negative Speech Enhancement: Baseline (ONMF-Frobenius) vs. Comparator 1 (ONMF-KL) vs. Comparator 2 (NMF-HMM-KL)",
            ha="center", va="center", fontsize=9.8, fontstyle="italic", color="#475569")

    def draw_box(x, y, w, h, title, subtitle="", fill_color="#FFFFFF", edge_color="#CBD5E1", title_color="#0F172A", corner=0.03, lw=1.5):
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle=f"round,pad=0.05,rounding_size={corner}",
                                     facecolor=fill_color, edgecolor=edge_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x, y + h*0.18, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color, zorder=3)
            ax.text(x, y - h*0.18, subtitle, ha="center", va="center", fontsize=7.8, color="#475569", zorder=3)
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color, zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#64748B", style="->", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3), zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.16, label, ha="center", va="center", fontsize=7.8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=c_bg, edgecolor="none"), zorder=5)

    # 1. Inputs (fs = 48 kHz)
    draw_box(3.0, 9.40, 4.4, 0.72, "Clean Speech Prior (Training)", "whitman.wav | S_sp = |STFT(s)| (fs = 48 kHz)",
             fill_color=c_blue_fill, edge_color=c_blue_border, title_color="#1E40AF")
    draw_box(8.0, 9.40, 4.4, 0.72, "Noise Prior Reference (Training)", "WhiteNoise.wav / AirNoise.wav | S_n = |STFT(n)|",
             fill_color=c_gray_fill, edge_color=c_gray_border, title_color="#334155")
    draw_box(13.0, 9.40, 4.4, 0.72, "Noisy Test Mixture (Evaluation)", "Shakes_White / Shakes_Air | Y = STFT(y)",
             fill_color=c_amber_fill, edge_color=c_amber_border, title_color="#92400E")

    # 2. Dictionary / Acoustic State Learning
    draw_box(3.0, 8.20, 4.4, 0.68, "Speech Dictionary / HMM Learning", "ONMF Frobenius | ONMF-KL | Baum-Welch EM (30 iters)",
             fill_color="#FFFFFF", edge_color=c_blue_border, lw=1.3)
    draw_box(8.0, 8.20, 4.4, 0.68, "Noise Dictionary / HMM Learning", "ONMF Frobenius | ONMF-KL | Multi-State Noise HMM",
             fill_color="#FFFFFF", edge_color=c_gray_border, lw=1.3)
    draw_arrow(3.0, 9.04, 3.0, 8.54, color=c_blue_border)
    draw_arrow(8.0, 9.04, 8.0, 8.54, color=c_gray_border)

    # 3. Learned Dictionaries
    draw_box(3.0, 7.12, 4.2, 0.62, "Speech Basis W_s / HMM (W_j, A_s)", "Default: K_s=50 or (4x20) | Tuned: K_s=80 or (8x25)",
             fill_color=c_blue_fill, edge_color=c_blue_border, title_color="#1E40AF")
    draw_box(8.0, 7.12, 4.2, 0.62, "Noise Basis W_n / HMM (W_k, A_n)", "Default: K_n=10 or (2x10) | Tuned: K_n=20 or (4x20)",
             fill_color=c_gray_fill, edge_color=c_gray_border, title_color="#334155")
    draw_arrow(3.0, 7.86, 3.0, 7.43, color=c_blue_border)
    draw_arrow(8.0, 7.86, 8.0, 7.43, color=c_gray_border)

    # 4. Joint Supervised Basis Dispatcher
    draw_box(5.5, 6.12, 7.6, 0.60, "Joint Supervised Non-Negative Model [ W_speech | W_noise ]",
             "STFT Grid: Default nperseg=2048 (42.7 ms)  |  Parameter-Tuned nperseg=4096 (85.3 ms) + Apples-to-Apples Control",
             fill_color="#FFFFFF", edge_color="#475569", lw=1.6)
    draw_arrow(3.0, 6.81, 4.5, 6.42, color=c_blue_border)
    draw_arrow(8.0, 6.81, 6.5, 6.42, color=c_gray_border)
    draw_arrow(13.0, 9.04, 13.0, 5.82, label="Mixture STFT Y", color=c_amber_border)

    def draw_method_card(x, y, w, h, title, subtitle, bullets, fill_color, border_color):
        card = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                      boxstyle="round,pad=0.05,rounding_size=0.04",
                                      facecolor=fill_color, edgecolor=border_color, linewidth=1.8, zorder=2)
        ax.add_patch(card)
        hb = patches.FancyBboxPatch((x - w/2, y + h/2 - 0.72), w, 0.72,
                                    boxstyle="round,pad=0.03,rounding_size=0.02",
                                    facecolor=border_color, edgecolor=border_color, zorder=3)
        ax.add_patch(hb)
        ax.text(x, y + h/2 - 0.23, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#FFFFFF", zorder=4)
        ax.text(x, y + h/2 - 0.51, subtitle, ha="center", va="center", fontsize=7.6, fontstyle="italic", color="#F8FAFC", zorder=4)
        sy = y + h/2 - 1.00
        for b in bullets:
            ax.text(x - w/2 + 0.18, sy, b, ha="left", va="center", fontsize=7.7, color="#1E293B", zorder=4)
            sy -= 0.36

    card_w = 4.4
    card_h = 2.85
    card_y = 3.92
    xs = [3.0, 8.0, 13.0]

    draw_method_card(xs[0], card_y, card_w, card_h,
                     "1. BASELINE", "ONMF + Frobenius (Sack et al., CISS 2022)",
                     ["- Objective: Euclidean / L2 Frobenius norm",
                      "- Dictionary: Online patch streaming",
                      "- Inference: Row-wise PGD (alpha = 100)",
                      "- Mask: Linear Wiener magnitude ratio",
                      "- Dual-track role: Def (2048) & Ctrl (4096)"],
                     c_blue_fill, c_blue_border)

    draw_method_card(xs[1], card_y, card_w, card_h,
                     "2. COMPARATOR 1", "ONMF-KL (EM/MM Audited, Future Work)",
                     ["- Objective: Scale-invariant Gen. KL Div.",
                      "- EM/MM stats: A_t = (1-g)A + g(W*(X/WH)H^T)",
                      "- Inference: 50 Multiplicative Update iters",
                      "- Tuned: nperseg=4096, K=[80,20], alpha_s=0.10",
                      "- Fast inference & sparse vocal protection"],
                     c_amber_fill, c_amber_border)

    draw_method_card(xs[2], card_y, card_w, card_h,
                     "3. COMPARATOR 2", "NMF-HMM-KL (Xiang et al., EURASIP 2022)",
                     ["- Dynamic HMM: J_s states x J_n noise states",
                      "- Kronecker: pi_s (x) pi_n and A_s (x) A_n",
                      "- Scaled float STFT + causal posterior w_n",
                      "- Mask: Expected MMSE spectral gain (Eq. 46)",
                      "- Tuned: 4096, arch=(8,25,4,20), alpha_s=0.05"],
                     c_green_fill, c_green_border)

    for x_s, col in zip(xs, [c_blue_border, c_amber_border, c_green_border]):
        draw_arrow(5.5, 5.82, x_s - 0.15, 5.38, color=col)
    for x_s in xs:
        draw_arrow(13.0, 5.82, x_s + 0.15, 5.38, color=c_amber_border)

    # 5. Time-Domain Synthesis
    draw_box(8.0, 1.92, 14.4, 0.68,
             "Time-Domain Waveform Synthesis (Synthesis Filterbank)",
             "Estimated Speech Magnitude |S_hat| + Mixed Noisy Phase exp(j * angle(Y))  ->  Overlap-Add iSTFT  ->  s_hat(t)",
             fill_color=c_purple_fill, edge_color=c_purple_border, title_color="#5B21B6", corner=0.03)

    for x_c, col in zip(xs, [c_blue_border, c_amber_border, c_green_border]):
        draw_arrow(x_c, 2.49, x_c, 2.26, color=col)

    # 6. Dual-Track Evaluation
    draw_box(8.0, 0.78, 14.4, 0.80,
             "DUAL-TRACK STATISTICAL EVALUATION: DEFAULT AUDITED (2048) & PARAMETER-TUNED AIR NOISE (4096)",
             "BSS Eval (SDR, SIR, SAR [dB])  |  PESQ WB (48kHz -> 16kHz)  |  STOI  |  VAD F1  |  5-Fold Friedman + Holm Paired t-Test + Pearson r",
             fill_color="#FFFFFF", edge_color="#0F172A", title_color="#0F172A", corner=0.03, lw=2.0)
    draw_arrow(8.0, 1.58, 8.0, 1.18, color="#0F172A")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] Architecture diagram saved to:", output_path)


if __name__ == "__main__":
    create_architecture_diagram()
