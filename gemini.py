import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_periodic_noise(image_path, noise_type='grid', dc_radius=20, block_size=5, h_blocksize=5, v_blocksize=5):
    """
    依照自訂遮罩去除影像中的週期性雜訊
    :param noise_type: 'grid' (網格雜訊) 或 'horizontal' (水平雜訊)
    :param notch_points: 網格雜訊的亮點偏移座標清單 (dx, dy)
    :param dc_radius: 中心 DC 成分的保留半徑 (像素)
    :param block_size: 遮罩區塊/線條的寬度 (像素)
    """
    
    # 【步驟 1】影像讀取與前處理
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("無法讀取影像，請確認路徑是否正確。")
        
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # 取得中心點座標

    # 【步驟 2】空間域轉換至頻率域 (FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 將低頻 DC 移至矩陣中心
    
    # 計算頻譜對數值 (僅供視覺化觀察，不參與後續計算)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-1)

    # 【步驟 3】設計與應用陷波濾波器遮罩 (Notch Filter Mask)
    mask = np.ones((rows, cols), dtype=np.float32)

    if noise_type == 'horizontal':
        # 針對圖 8 水平雜訊：水平雜訊會在頻譜的「垂直中軸線」上產生高頻亮點
        # 遮擋垂直中軸線 (設為 0)
        mask[:, ccol-block_size : ccol+block_size+1] = 0
        
        # 關鍵：保留正中心 DC 區域 (畫上全為 1 的圓形)
        cv2.circle(mask, (ccol, crow), dc_radius, 1, thickness=-1)
    elif noise_type == 'vertical':
        # 針對垂直雜訊 (空間域直線)：遮擋頻譜的「水平中軸線」
        # 注意這裡操作的是 row (y軸) 的切片
        mask[crow-block_size : crow+block_size+1, :] = 0
        
        # 關鍵：保留正中心 DC 區域
        cv2.circle(mask, (ccol, crow), dc_radius, 1, thickness=-1)

    elif noise_type == 'cross':
    # 步驟一：先做所有「破壞 (設為 0)」的動作
        mask[:, ccol-v_blocksize : ccol+v_blocksize+1] = 0  # 畫垂直黑條
        mask[crow-h_blocksize : crow+h_blocksize+1, :] = 0  # 畫水平黑條
        
        # 步驟二：最後做「保護 (設為 1)」的動作，確保中心絕對不會被剛剛的黑線蓋住
        cv2.circle(mask, (ccol, crow), dc_radius, 1, thickness=-1)
    # elif noise_type == 'grid':
    #     # 針對圖 7 網格雜訊：在特定座標畫上黑色小圓形阻擋
    #     if notch_points is None:
    #         # 這裡需要根據你實際頻譜圖的亮點位置來設定 (相對於中心的 x, y 偏移量)
    #         # 舉例：右下與右上偏移 40 像素的亮點
    #         notch_points = [(40, 40), (40, -40)] 
            
    #     for dx, dy in notch_points:
    #         # 頻譜具有「共軛對稱性」，所以打點時必須對稱阻擋 (中心點 ± 偏移量)
    #         cv2.circle(mask, (ccol + dx, crow + dy), block_size, 0, thickness=-1)
    #         cv2.circle(mask, (ccol - dx, crow - dy), block_size, 0, thickness=-1)

    # 【步驟 4】頻域濾波 (Element-wise multiplication)
    fshift_filtered = fshift * mask

    # 供視覺化觀察濾波後的頻譜
    magnitude_filtered = 20 * np.log(np.abs(fshift_filtered) + 1e-1)

    # 【步驟 5】頻率域轉換回空間域 (IFFT)
    f_ishift = np.fft.ifftshift(fshift_filtered) # 將中心點移回原點
    img_back = np.fft.ifft2(f_ishift)            # 二維反傅立葉轉換

    # 【步驟 6】影像還原與輸出
    img_back = np.abs(img_back)                  # 取絕對值 (Magnitude)
    
    # 正規化 (Normalize) 至 0-255 範圍
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_clean = np.uint8(img_back)               # 轉換資料型態為 uint8

    # --- Matplotlib 繪圖與視覺化 ---
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('1. Original Image'), plt.axis('off')
    
    plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('2. FFT Spectrum'), plt.axis('off')
    
    plt.subplot(2, 3, 3), plt.imshow(mask, cmap='gray')
    plt.title('3. Notch Mask'), plt.axis('off')
    
    plt.subplot(2, 3, 4), plt.imshow(magnitude_filtered, cmap='gray')
    plt.title('4. Filtered Spectrum'), plt.axis('off')
    
    plt.subplot(2, 3, 5), plt.imshow(img_clean, cmap='gray')
    plt.title('5. Cleaned Result'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return img_clean

# ---------------- 使用範例 ----------------
# 處理水平雜訊 (圖 8)
# clean_img_h = remove_periodic_noise('yzu8noiseg.png', noise_type='horizontal', dc_radius=20, block_size=2)
# cv2.imwrite("yzu8noiseg_denoised.png", clean_img_h)

# 處理網格雜訊 (圖 7)
# 假設你觀察到亮點在距離中心 (x=50, y=30) 以及 (x=50, y=-30) 的位置
clean_img_g = remove_periodic_noise('yzu7noiseg.png', noise_type='cross', dc_radius=25, h_blocksize=6, v_blocksize=5)
cv2.imwrite("yzu7noiseg_denoised_2.png", clean_img_g)