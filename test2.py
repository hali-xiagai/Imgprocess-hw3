"""
gaussian_notch_filter.py
========================
使用 Gaussian Notch Reject Filter 消除影像中的線條 / 週期性雜訊。

原理
----
1. FFT → 頻率域
2. 在振幅頻譜上自動偵測雜訊峰值（軸上 + 離軸）
3. 對每個雜訊點建立 Gaussian Notch（凹洞），濾波器 H(u,v) = 乘積
4. 逆 FFT 還原影像，並補償亮度/對比

兩種 Notch 類型
---------------
* **Axis Notch**（帶阻）：針對位於頻譜主軸上的峰值，用高斯帶阻濾除整條軸
  - 水平軸 (u=0) → 消除原圖的「垂直」條紋
  - 垂直軸 (v=0) → 消除原圖的「水平」條紋
* **Point Notch**：針對離軸的孤立亮點，精準挖小洞

參數調整指南
-----------
protect_radius    : DC 中心保護半徑，影像越大/細節越豐富可調大（建議 60~100）
axis_notch_sigma  : 軸向帶阻的高斯寬度（越大消除越徹底但細節損失越多）
                    水平條紋圖：6~10；網格圖：8~14
axis_strength     : 軸向帶阻強度 [0~1]，< 1 可保留部分細節
                    建議 0.80~0.95
point_threshold   : 偵測離軸峰值的閾值（相對最大值的比例，0~1）
                    越低偵測越多，但可能誤殺
point_sigma       : 點 notch 的高斯寬度（建議 8~15）
dc_band_protect   : 在軸上距中心多少像素內不做帶阻（保留低頻資訊）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
#  核心濾波器
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_axis_notch(shape, axis="horizontal",
                         sigma=20,   # 🔥 原本 8 → 改 20
                         strength=0.98,  # 🔥 加強
                         dc_protect=40):  # 🔥 原本 80 → 改 40

    """
    沿一條主軸建立 Gaussian Band-Stop (帶阻) Notch。

    axis="horizontal" → 在 u=0 行做帶阻（消除原圖垂直條紋）
    axis="vertical"   → 在 v=0 列做帶阻（消除原圖水平條紋）

    H(u,v) = 1 - strength * exp(-D²/(2σ²)) * mask_dc

    其中 D 為到目標軸的距離，mask_dc 在 DC 保護區內強制 = 0（不挖）。
    """
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)

    if axis == "horizontal":
        # 距 u=0 軸的距離 = |u|；但保護 |v| < dc_protect 的中心區域不被挖
        D = np.abs(U).astype(np.float64)
        far = (np.abs(V) >= dc_protect).astype(np.float64)
    else:  # "vertical"
        D = np.abs(V).astype(np.float64)
        far = (np.abs(U) >= dc_protect).astype(np.float64)

    notch = 1.0 - strength * np.exp(-D ** 2 / (2.0 * sigma ** 2)) * far
    return notch


def gaussian_point_notch(shape, peaks, sigma=10):
    """
    對每個離軸峰值 (u0, v0) 及其對稱點 (-u0, -v0) 建立高斯凹洞。
    所有凹洞相乘得到最終 point notch filter。
    """
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)

    H = np.ones((rows, cols), dtype=np.float64)
    for (u0, v0) in peaks:
        D1_sq = (U - u0) ** 2 + (V - v0) ** 2
        D2_sq = (U + u0) ** 2 + (V + v0) ** 2
        notch = 1.0 - np.exp(-D1_sq / (2 * sigma ** 2)) * \
                       np.exp(-D2_sq / (2 * sigma ** 2))
        H *= notch
    return H


def detect_off_axis_peaks(mag_abs, center, dc_protect=80,
                           axis_band=12, point_threshold=0.3):
    """
    偵測離軸（不在主十字帶上）的雜訊峰值。

    Returns list of (u, v) in shifted coords（只回傳上半平面）。
    """
    rows, cols = mag_abs.shape
    cu, cv_c = center

    # 把 DC 保護區和兩條主軸帶遮蓋
    mask = mag_abs.copy().astype(np.float64)
    # DC center
    cv2.circle(mask, (cv_c, cu), dc_protect, 0, -1)
    # 主軸帶 (±axis_band)
    mask[cu - axis_band:cu + axis_band + 1, :] = 0
    mask[:, cv_c - axis_band:cv_c + axis_band + 1] = 0

    # 用振幅最大值比例做閾值
    peak_val = mask.max()
    if peak_val == 0:
        return []

    threshold = peak_val * point_threshold
    _, binary = cv2.threshold(mask.astype(np.float32), threshold, 255,
                               cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # 只看上半平面
    half = binary.copy()
    half[cu:, :] = 0

    contours, _ = cv2.findContours(half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peaks = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            ci = int(M["m01"] / M["m00"])
            cj = int(M["m10"] / M["m00"])
            peaks.append((ci - cu, cj - cv_c))
    return peaks


# ──────────────────────────────────────────────────────────────────────────────
#  主函式
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_notch_filter(
    image_path,
    # ── 軸向帶阻參數 ───────────────────────────────
    h_axis_sigma=10,        # 水平軸 (u=0) 帶阻寬度 → 消除原圖垂直條紋
    h_axis_strength=0.92,   # 水平軸帶阻強度
    v_axis_sigma=10,        # 垂直軸 (v=0) 帶阻寬度 → 消除原圖水平條紋
    v_axis_strength=0.92,   # 垂直軸帶阻強度
    dc_band_protect=80,     # 靠近 DC 中心、不施加帶阻的範圍（像素）
    # ── 離軸點 notch 參數 ──────────────────────────
    detect_off_axis=True,   # 是否偵測並消除離軸峰值
    point_threshold=0.35,   # 離軸峰值偵測閾值（相對最大值），越低偵測越多
    point_sigma=12,         # 離軸 notch 寬度
    # ── 其他 ────────────────────────────────────────
    protect_radius=80,      # DC 整體保護半徑
):
    """
    對影像執行完整的 Gaussian Notch Reject Filter 流程。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot open: {image_path}")

    rows, cols = img.shape
    cu, cv_c = rows // 2, cols // 2

    # ── FFT ────────────────────────────────────────────────────────────────
    f        = np.fft.fft2(img.astype(np.float64))
    f_shift  = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.float64)

    # 視覺化用振幅頻譜
    mag_log  = 20 * np.log1p(mag)
    mag_vis  = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Step 1: 軸向 Gaussian Band-Stop Notch ──────────────────────────────
    H = np.ones((rows, cols), dtype=np.float64)

    H *= gaussian_axis_notch((rows, cols), axis="horizontal",
                              sigma=h_axis_sigma,
                              strength=h_axis_strength,
                              dc_protect=dc_band_protect)

    H *= gaussian_axis_notch((rows, cols), axis="vertical",
                              sigma=v_axis_sigma,
                              strength=v_axis_strength,
                              dc_protect=dc_band_protect)

    # ── Step 2: 離軸 Point Notch ───────────────────────────────────────────
    off_peaks = []
    if detect_off_axis:
        off_peaks = detect_off_axis_peaks(
            mag,
            (cu, cv_c),
            dc_protect=protect_radius,
            axis_band=max(h_axis_sigma, v_axis_sigma) + 4,
            point_threshold=point_threshold,
        )
        if off_peaks:
            H *= gaussian_point_notch((rows, cols), off_peaks, sigma=point_sigma)
            print(f"  Detected {len(off_peaks)} off-axis noise peaks (upper half)")
        else:
            print("  No significant off-axis peaks detected")

    # ── Step 3: 保護 DC 中心（確保不被誤改）──────────────────────────────
    # 在 DC 區域強制 H=1（保留所有低頻資訊）
    rr, cc = np.ogrid[-cu:rows - cu, -cv_c:cols - cv_c]
    dc_mask = (rr ** 2 + cc ** 2) <= protect_radius ** 2
    # H_smooth[dc_mask] = 1.0
    H[dc_mask] = 1.0

    # ── Step 4: 輕微平滑濾波器邊緣 ────────────────────────────────────────
    H_smooth = cv2.GaussianBlur(H.astype(np.float32), (7, 7), 0).astype(np.float64)
    # 再次保護 DC（blur 可能擴散到 DC 邊緣）
    H_smooth[dc_mask] = 1.0

    # ── Step 5: 套用並逆 FFT ───────────────────────────────────────────────
    filtered_shift = f_shift * H_smooth
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_shift)))

    # ── Step 6: 亮度/對比保留（線性對齊到原圖統計量）─────────────────────
    orig_mean, orig_std = float(img.mean()), float(img.std())
    rec_mean,  rec_std  = img_back.mean(), img_back.std()

    if rec_std > 1e-6:
        img_back = (img_back - rec_mean) * (orig_std / rec_std) + orig_mean

    img_fixed = np.clip(img_back, 0, 255).astype(np.uint8)
    print("H min/max:", H_smooth.min(), H_smooth.max())
    print("Mask coverage:", np.mean(H_smooth < 0.5))

    return img, img_fixed, mag_vis, H_smooth, off_peaks


# ──────────────────────────────────────────────────────────────────────────────
#  視覺化與輸出
# ──────────────────────────────────────────────────────────────────────────────

def visualize_and_save(img_orig, img_fixed, mag_vis, H, off_peaks,
                        out_fixed_path, title=""):
    """四格比較圖 + 儲存修復影像。"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Microsoft JhengHei',
                                        'Taipei Sans TC Beta', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rows, cols = img_orig.shape
    cu, cv_c = rows // 2, cols // 2

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.12)

    # Panel 1 – 原圖
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Original (with noise)", fontsize=11)
    ax1.axis('off')

    # Panel 2 – 振幅頻譜 + 偵測到的峰值
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mag_vis, cmap='gray')
    for (u0, v0) in off_peaks:
        ax2.plot(cv_c + v0, cu + u0, 'r+', ms=10, mew=1.8)
        ax2.plot(cv_c - v0, cu - u0, 'b+', ms=10, mew=1.8)
    n = len(off_peaks)
    ax2.set_title(f"Amplitude Spectrum  (detected {n*2} off-axis peaks)", fontsize=10)
    ax2.axis('off')

    # Panel 3 – Notch Filter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(H, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("Gaussian Notch Filter  H(u,v)\n(Black = suppressed frequencies)", fontsize=10)
    ax3.axis('off')

    # Panel 4 – 修復後
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img_fixed, cmap='gray', vmin=0, vmax=255)
    ax4.set_title("Restored Image", fontsize=11)
    ax4.axis('off')

    # 儲存
    cv2.imwrite(out_fixed_path, img_fixed)
    vis_path = out_fixed_path.replace('.png', '_vis.png')
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fixed_path}")
    print(f"  Saved: {vis_path}")
    return vis_path


# ──────────────────────────────────────────────────────────────────────────────
#  執行區塊
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ════════════════════════════════════════════════════════════════════════
    # 圖1: yzu7noiseg.png — 嚴重網格狀（水平 + 垂直）條紋雜訊
    # ════════════════════════════════════════════════════════════════════════
    # 網格雜訊的頻譜峰值同時出現在水平軸 (v~±101) 和垂直軸 (u~±100)
    # 需要同時對兩軸做帶阻，且 sigma 要夠大才能覆蓋多個諧波
    print("=== Image 1: Grid noise (yzu7noiseg.png) ===")
    img1, fix1, mag1, H1, peaks1 = gaussian_notch_filter(
        "yzu7noiseg.png",
        h_axis_sigma     = 14,     # 水平軸帶阻寬度（消除垂直條紋）
        h_axis_strength  = 0.95,   # 強力消除
        v_axis_sigma     = 14,     # 垂直軸帶阻寬度（消除水平條紋）
        v_axis_strength  = 0.98,
        dc_band_protect  = 90,     # 靠近 DC 不挖洞
        detect_off_axis  = True,
        point_threshold  = 0.30,   # 偵測更多離軸峰值
        point_sigma      = 14,
        protect_radius   = 70,
    )
    visualize_and_save(
        img1, fix1, mag1, H1, peaks1,
        "yzu7noiseg_fixed.png",
        title="Image 1 – Grid Noise  |  Gaussian Notch Reject Filter",
    )

    # ════════════════════════════════════════════════════════════════════════
    # 圖2: yzu8noiseg.png — 水平條紋雜訊
    # ════════════════════════════════════════════════════════════════════════
    # 水平條紋 → 頻譜垂直軸 (v=0) 有強峰值
    # 不需要對水平軸做強帶阻（否則會消除建築物的垂直邊緣細節）
    # v_axis_strength 稍高以徹底消除水平線
    print("\n=== Image 2: Horizontal stripe noise (yzu8noiseg.png) ===")
    img2, fix2, mag2, H2, peaks2 = gaussian_notch_filter(
    "yzu8noiseg.png",

    h_axis_sigma     = 10,
    h_axis_strength  = 0.75,

    v_axis_sigma     = 30,   # 🔥 關鍵：拉大
    v_axis_strength  = 0.99, # 🔥 幾乎全刪

    dc_band_protect  = 40,   # 🔥 原本 85 → 太大

    detect_off_axis  = True,
    point_threshold  = 0.20, # 🔥 更敏感
    point_sigma      = 18,

    protect_radius   = 60,
)

    visualize_and_save(
        img2, fix2, mag2, H2, peaks2,
        "yzu8noiseg_fixed.png",
        title="Image 2 – Horizontal Stripe Noise  |  Gaussian Notch Reject Filter",
    )

    print("\n Done!")