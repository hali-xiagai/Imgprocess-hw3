import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_remove_line_noise(image_path, protect_radius=30, noise_threshold=180):
    """
    全自動偵測並消除影像中的線條/週期性雜訊
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
         raise ValueError("找不到影像，請確認路徑。")

    rows, cols = img.shape
    center_u, center_v = rows // 2, cols // 2

    # 1. 轉換到頻率域並計算振幅頻譜 (壓縮到 0-255 以便做影像處理)
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    mag_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    mag_normalized = cv2.normalize(mag_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    mag_uint8 = np.uint8(mag_normalized)

    # ==========================================
    # 開始自動製作濾波遮罩 (Mask)
    # ==========================================
    
    # 2. 自動找出異常亮點 (透過閥值化)
    # 大於 noise_threshold 的地方會變成 255 (白色)，其他變 0 (黑色)
    _, noise_mask = cv2.threshold(mag_uint8, noise_threshold, 255, cv2.THRESH_BINARY)

    # 3. 保護中心區域 (強迫中心區域的雜訊遮罩為 0)
    # 避免誤刪影像的主要低頻結構
    cv2.circle(noise_mask, (center_v, center_u), protect_radius, 0, -1)

    # 設定要強制清除的水平線寬度 (視你的雜訊粗細而定，通常 3~7)
    slit_width_v = 8
    slit_width_u = 9
    # 設定極小核心保護區 (絕對不能挖掉正中心，否則圖片會變全灰)
    core_dc_radius = 50

    # --- 第一刀：挖除水平軸 (消除原圖中的「垂直」雜訊線) ---
    for v in range(cols):
        if abs(v - center_v) > core_dc_radius:
            noise_mask[center_u - slit_width_v : center_u + slit_width_v + 1, v] = 255

    # --- 第二刀：挖除垂直軸 (消除原圖中的「水平」雜訊線) ---
    for u in range(rows):
        if abs(u - center_u) > core_dc_radius:
            noise_mask[u, center_v - slit_width_u : center_v + slit_width_u + 1] = 255
            
    # 4. 膨脹雜訊點 (Dilate)
    # 因為線條可能會有斷點或很細，我們用一個 5x5 的方塊把它變粗，確保雜訊被完全覆蓋
    # kernel = np.ones((5, 5), np.uint8)
    kernel = np.ones((3, 3), np.uint8)  # 原本 5x5
    iterations = 1                      # 原本 2

    noise_mask_thick = cv2.dilate(noise_mask, kernel, iterations=2)

    # 5. 製作最終的濾波器 H (反轉顏色：雜訊處為 0阻擋，其餘為 1通過)
    # 雜訊處原本是 255，除以 255 變成 1。用 1 去減，就變成 0 (黑色/挖洞)。
    
    H = 1.0 - (noise_mask_thick / 255.0)

    # 6. 羽化邊緣 (Gaussian Blur)
    # 讓洞的邊緣平滑過渡，這一步等同於「高斯濾波」，可避免還原後產生水波紋
    H_blurred = H #cv2.GaussianBlur(H, (21, 21), 0)

    # ==========================================
    # 套用濾波器並還原
    # ==========================================

    # 7. 將頻譜乘上自動生成的遮罩
    filtered_shift = f_shift * H_blurred

    # 8. 反向轉換回空間域
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    img_reconstructed = np.real(img_back)
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)

    return img, img_reconstructed, mag_uint8, H_blurred

# 執行區塊
if __name__ == "__main__":
    IMAGE_PATH = 'yzu7noiseg.png' # 替換成你的圖片
    
    # 你只需要微調這兩個參數：
    # protect_radius: 影像主體越大、對比越強，這個圓要設越大 (預設 30-50)
    # noise_threshold: 如果雜訊沒被濾乾淨就調低 (例如 150)；如果原圖細節被吃掉就調高 (例如 200)
    PROTECT_RADIUS = 70 
    NOISE_THRESHOLD = 200

    try:
        img_orig, img_fixed, mag_vis, auto_mask = auto_remove_line_noise(IMAGE_PATH, PROTECT_RADIUS, NOISE_THRESHOLD)

        # 畫圖視覺化
        plt.figure(figsize=(16, 10))
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC Beta', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(2, 2, 1), plt.imshow(img_orig, cmap='gray'), plt.title('原始影像 (含線條雜訊)')
        plt.subplot(2, 2, 2), plt.imshow(mag_vis, cmap='gray'), plt.title('原圖振幅頻譜')
        
        # 顯示自動生成的遮罩，越黑代表該處頻率被阻擋得越徹底
        plt.subplot(2, 2, 3), plt.imshow(auto_mask, cmap='gray'), plt.title('自動生成的羽化遮罩 H(u,v)')
        plt.subplot(2, 2, 4), plt.imshow(img_fixed, cmap='gray'), plt.title('修復後的影像')
        cv2.imwrite(f"fixed_{IMAGE_PATH}", img_fixed)
        for i in range(1, 5):
            plt.subplot(2, 2, i).axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)