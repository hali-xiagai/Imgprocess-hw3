import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_fourier_spectrums(image_path):
    """
    讀取影像並回傳原圖、振幅頻譜與相位頻譜
    """
    # 1. 讀取影像並轉為灰階 (傅立葉轉換通常在單一通道上進行)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"找不到影像：{image_path}，請確認路徑是否正確。")

    # 2. 進行二維快速傅立葉轉換 (2D FFT)
    f_transform = np.fft.fft2(img)
    
    # 3. 頻譜位移 (Shift)：將低頻(原點)從左上角移到影像中心，方便觀察
    f_shift = np.fft.fftshift(f_transform)

    # 4. 計算振幅頻譜 (Amplitude Spectrum)
    # 取絕對值。因為高低頻的數值差異極大，我們加上 1 並取 log (對數)，
    # 這樣才能將數據壓縮到人眼可視的灰階範圍內。
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # 5. 計算相位頻譜 (Phase Spectrum)
    # 取複數的角度 (結果會介於 -π 到 π 之間)
    phase_spectrum = np.angle(f_shift)
    # 1. 將數值按比例縮放 (Normalize) 到 0 ~ 255 的範圍
    normalized_mag = cv2.normalize(magnitude_spectrum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_spe = cv2.normalize(phase_spectrum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
# 2. 正式將資料型態轉換為 8-bit 無號整數 (uint8)
    mag_uint8 = np.uint8(normalized_mag)
    spe_uint8 = np.uint8(normalized_spe)
    cv2.imwrite(f"magnitude_spectrum_{image_path}", mag_uint8)
    cv2.imwrite(f"phase_spectrum_{image_path}", spe_uint8)
    return img, magnitude_spectrum, phase_spectrum

def plot_spectrums(img1_path, img2_path):
    """
    處理兩張影像並繪製對比圖
    """
    # 取得兩張圖片的頻譜資料
    img1, mag1, phase1 = get_fourier_spectrums(img1_path)
    img2, mag2, phase2 = get_fourier_spectrums(img2_path)

    # 設定繪圖視窗大小
    plt.figure(figsize=(15, 8))
    
    # 設定字體，避免 matplotlib 中文顯示亂碼 (視作業系統而定，此為通用設定)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC Beta', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 整理資料以便用迴圈畫圖
    images = [img1, mag1, phase1, img2, mag2, phase2]
    titles = ['影像 A (原圖)', '影像 A (振幅頻譜)', '影像 A (相位頻譜)',
              '影像 B (原圖)', '影像 B (振幅頻譜)', '影像 B (相位頻譜)']

    # 繪製 2x3 的子圖表
    for i in range(6):
        plt.subplot(2, 3, i+1)
        # 相位頻譜的值域包含負數，使用 gray colormap 時會自動 mapping 到 0-255 視覺化
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off') # 隱藏座標軸

    plt.tight_layout()
    plt.show()

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    # 請將這裡替換成你電腦中實際的圖片路徑
    # 建議準備兩張特徵不同的圖 (例如：一張風景大樓照、一張人臉特寫照)
    IMAGE_1_PATH = 'yzu7noiseg.png' 
    IMAGE_2_PATH = 'yzu8noiseg.png'

    try:
        plot_spectrums(IMAGE_1_PATH, IMAGE_2_PATH)
    except Exception as e:
        print(e)