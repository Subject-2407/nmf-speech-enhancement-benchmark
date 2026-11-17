import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_architecture_diagram(output_path="Audio-Enhancement-via-ONMF/pembanding/architecture_flowchart.png"):
    fig = plt.figure(figsize=(20, 12), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    c_bg = "#F8FAFC"
    c_title = "#0F172A"
    c_blue_border, c_blue_fill = "#2563EB", "#EFF6FF"
    c_amber_border, c_amber_fill = "#D97706", "#FFFBEB"
    c_green_border, c_green_fill = "#059669", "#ECFDF5"
    c_rose_border, c_rose_fill = "#E11D48", "#FFF1F2"
    c_teal_border, c_teal_fill = "#0D9488", "#F0FDFA"
    c_indigo_border, c_indigo_fill = "#4F46E5", "#EEF2FF"
    c_purple_border, c_purple_fill = "#7C3AED", "#F5F3FF"
    c_gray_border, c_gray_fill = "#64748B", "#F1F5F9"

    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    ax.text(10, 11.45, "ARSITEKTUR & ALIRAN EKSPERIMEN KOMPARATIF AUDIO ENHANCEMENT",
            ha="center", va="center", fontsize=15, fontweight="bold", color=c_title)
    ax.text(10, 11.08,
            "Supervised NMF (Baseline, ONMF-KL, NMF-HMM-KL) vs. Unsupervised Zero-Prior (IWF, WNNM-RPCA, WSNM-RPCA)",
            ha="center", va="center", fontsize=9.8, fontstyle="italic", color="#475569")

    def draw_box(x, y, w, h, title, subtitle="", fill_color="#FFFFFF", edge_color="#CBD5E1", title_color="#0F172A", corner=0.03, lw=1.5):
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle=f"round,pad=0.05,rounding_size={corner}",
                                     facecolor=fill_color, edgecolor=edge_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x, y + h*0.18, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color=title_color, zorder=3)
            ax.text(x, y - h*0.18, subtitle, ha="center", va="center", fontsize=7.6, color="#475569", zorder=3)
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color=title_color, zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#64748B", style="->", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3), zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.16, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=c_bg, edgecolor="none"), zorder=5)

    # 1. Inputs
    draw_box(3.5, 10.15, 4.4, 0.72, "Prior Wicara Bersih (Supervised)", "whitman.wav | S_sp = |STFT(s)|", fill_color=c_blue_fill, edge_color=c_blue_border, title_color="#1E40AF")
    draw_box(8.8, 10.15, 4.4, 0.72, "Prior Derau Acuan (Supervised)", "WhiteNoise / AirNoise | S_n = |STFT(n)|", fill_color=c_gray_fill, edge_color=c_gray_border, title_color="#334155")
    draw_box(15.6, 10.15, 5.2, 0.72, "Sinyal Campuran Uji (All Methods)", "Shakes_White / Shakes_Air | Y = STFT(y)", fill_color=c_amber_fill, edge_color=c_amber_border, title_color="#92400E")

    # 2. Dict Learning (for Supervised methods)
    draw_box(3.5, 8.9, 4.4, 0.70, "Pembelajaran Kamus Wicara", "ONMF Frobenius / ONMF-KL / HMM EM", fill_color="#FFFFFF", edge_color=c_blue_border, lw=1.2)
    draw_box(8.8, 8.9, 4.4, 0.70, "Pembelajaran Kamus Derau", "ONMF / KL-NMF / Multi-State Noise HMM", fill_color="#FFFFFF", edge_color=c_gray_border, lw=1.2)
    draw_arrow(3.5, 9.79, 3.5, 9.25)
    draw_arrow(8.8, 9.79, 8.8, 9.25)

    # 3. Dictionaries
    draw_box(3.5, 7.75, 4.0, 0.65, "Kamus Wicara W_speech", "K_s = 50 (atau J=4 x 20 atom)", fill_color=c_blue_fill, edge_color=c_blue_border, title_color="#1E40AF")
    draw_box(8.8, 7.75, 4.0, 0.65, "Kamus Derau W_noise", "K_n = 10 (atau J_n=2 x 10 atom)", fill_color=c_gray_fill, edge_color=c_gray_border, title_color="#334155")
    draw_arrow(3.5, 8.55, 3.5, 8.08)
    draw_arrow(8.8, 8.55, 8.8, 8.08)

    # 4. Combined Dict + Zero-prior split
    draw_box(6.15, 6.70, 6.2, 0.62, "Kamus Gabungan Supervised [ W_speech | W_noise ]", "Basis Apriori Sinyal Wicara & Interferensi Derau", fill_color="#FFFFFF", edge_color="#475569", lw=1.6)
    draw_arrow(3.5, 7.42, 4.8, 7.01)
    draw_arrow(8.8, 7.42, 7.5, 7.01)

    draw_box(15.6, 6.70, 5.4, 0.62, "Jalur Unsupervised (Zero-Prior)", "Tanpa Data Latih: Hanya Input Uji M = |STFT(y)|", fill_color="#FFF7ED", edge_color=c_amber_border, title_color="#9A3412", lw=1.6)
    draw_arrow(15.6, 9.79, 15.6, 7.01, label="Input Uji Y", color=c_amber_border)
    draw_arrow(12.9, 6.70, 9.25, 6.70, color=c_amber_border)

    def draw_method_card(x, y, w, h, title, subtitle, bullets, fill_color, border_color):
        card = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                      boxstyle="round,pad=0.05,rounding_size=0.04",
                                      facecolor=fill_color, edgecolor=border_color, linewidth=1.7, zorder=2)
        ax.add_patch(card)
        hb = patches.FancyBboxPatch((x - w/2, y + h/2 - 0.72), w, 0.72,
                                    boxstyle="round,pad=0.03,rounding_size=0.02",
                                    facecolor=border_color, edgecolor=border_color, zorder=3)
        ax.add_patch(hb)
        ax.text(x, y + h/2 - 0.24, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF", zorder=4)
        ax.text(x, y + h/2 - 0.52, subtitle, ha="center", va="center", fontsize=6.8, fontstyle="italic", color="#F8FAFC", zorder=4)
        sy = y + h/2 - 0.98
        for b in bullets:
            ax.text(x - w/2 + 0.13, sy, b, ha="left", va="center", fontsize=6.7, color="#1E293B", zorder=4)
            sy -= 0.34

    card_w = 3.05
    card_h = 2.85
    card_y = 4.25
    xs = [1.85, 5.10, 8.35, 11.65, 14.90, 18.15]

    draw_method_card(xs[0], card_y, card_w, card_h,
                     "1. BASELINE", "ONMF + Frobenius (Sack 2022)",
                     ["• Supervised NMF (L2 Norm)",
                      "• Online Patch Streaming",
                      "• Projected Grad Descent",
                      "• Sparsitas alpha = 100",
                      "• Wiener Magnitude Mask"],
                     c_blue_fill, c_blue_border)

    draw_method_card(xs[1], card_y, card_w, card_h,
                     "2. PEMBANDING 1", "ONMF-KL Fixed (Sack FW)",
                     ["• Supervised Online KL-NMF",
                      "• Statistik EM/MM Cukup",
                      "• A_t = (1-g)A + g(W*(X/WH)H^T)",
                      "• Penalti Skala Relatif (KL)",
                      "• KL Wiener Ratio Mask"],
                     c_amber_fill, c_amber_border)

    draw_method_card(xs[2], card_y, card_w, card_h,
                     "3. PEMBANDING 2", "NMF-HMM-KL (Xiang 2022)",
                     ["• Supervised HMM Speech+Noise",
                      "• Kronecker: pi_s(x)pi_n, A_s(x)A_n",
                      "• Baum-Welch EM (30 iter)",
                      "• Posterior Forward omega_n",
                      "• MMSE Gain (Eq. 46)"],
                     c_green_fill, c_green_border)

    draw_method_card(xs[3], card_y, card_w, card_h,
                     "4. PEMBANDING 3", "Implicit Wiener (Jaiswal 2022)",
                     ["• Unsupervised Statistical Filter",
                      "• 25 ms Hamming, 50% overlap",
                      "• Recursive Noise (alpha=0.90)",
                      "• gamma = 4 - 0.15*SNRseg",
                      "• Super Cepat (<0.3s, Edge)"],
                     c_rose_fill, c_rose_border)

    draw_method_card(xs[4], card_y, card_w, card_h,
                     "5. METODE BARU A", "WNNM-RPCA (Sridhar 2022)",
                     ["• Unsupervised Low-Rank+Sparse",
                      "• Weighted Nuclear Norm (p=1)",
                      "• Bobot w_i = C*sqrt(mn)/(sv+eps)",
                      "• ADMM / IALM + Weighted SVT",
                      "• Tanpa Kamus / Tanpa VAD"],
                     c_teal_fill, c_teal_border)

    draw_method_card(xs[5], card_y, card_w, card_h,
                     "6. METODE BARU B", "WSNM-RPCA (Sridhar 2022)",
                     ["• Unsupervised Low-Rank+Sparse",
                      "• Weighted Schatten p-Norm",
                      "• Non-konveks p=0.7 (Eq. 12-14)",
                      "• ADMM + GST Thresholding",
                      "• Tanpa Kamus / Tanpa VAD"],
                     c_indigo_fill, c_indigo_border)

    # Arrows from Supervised & Unsupervised dispatchers
    for x_s, col in zip(xs[:3], [c_blue_border, c_amber_border, c_green_border]):
        draw_arrow(6.15, 6.39, x_s, 5.69, color=col)
    for x_u, col in zip(xs[3:], [c_rose_border, c_teal_border, c_indigo_border]):
        draw_arrow(15.6, 6.39, x_u, 5.69, color=col)

    # 6. Synthesis
    draw_box(10.0, 2.0, 14.5, 0.72,
             "Rekonstruksi Domain Waktu (Synthesis Filter)",
             "Spektrogram Magnitudo Terestimasi |S_hat| + Sudut Fase Campuran exp(j * angle(Y))  ->  iSTFT Overlap-Add  ->  Audio Bersih s_hat(t)",
             fill_color=c_purple_fill, edge_color=c_purple_border, title_color="#5B21B6", corner=0.03)

    for x_c, col in zip(xs, [c_blue_border, c_amber_border, c_green_border, c_rose_border, c_teal_border, c_indigo_border]):
        draw_arrow(x_c, 2.82, max(3.5, min(16.5, x_c)), 2.36, color=col)

    # 7. Evaluation
    draw_box(10.0, 0.78, 18.6, 0.80,
             "EVALUASI TERPADU DUAL-TRACK: DEFAULT TERAUDIT & PARAMETER-TUNED AIR NOISE (02_pembanding & 03_hasil_analisis)",
             "BSS Eval (SDR, SIR, SAR [dB])  |  PESQ WB (fs=48kHz -> 16kHz)  |  STOI  |  VAD F1  |  5-Fold + Friedman + Holm t-Test + Pearson r",
             fill_color="#FFFFFF", edge_color="#0F172A", title_color="#0F172A", corner=0.03, lw=2.0)
    draw_arrow(10.0, 1.64, 10.0, 1.18, color="#0F172A")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if "Audio-Enhancement-via-ONMF" not in output_path:
        plt.savefig(f"Audio-Enhancement-via-ONMF/{output_path}", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] Architecture diagram successfully saved to:", output_path)


if __name__ == "__main__":
    create_architecture_diagram()
