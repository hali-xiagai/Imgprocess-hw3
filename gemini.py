import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_periodic_noise(image_path, noise_type='horizontal', dc_radius=20, block_size=20, h_blocksize=5, v_blocksize=5, notch_points=None):
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
    elif noise_type == 'grid':
        # 針對圖 7 網格雜訊：在特定座標畫上黑色小圓形阻擋
        if notch_points is None:
            # 這裡需要根據你實際頻譜圖的亮點位置來設定 (相對於中心的 x, y 偏移量)
            # 舉例：右下與右上偏移 40 像素的亮點
            notch_points = []
        for dx, dy in notch_points:
            # 頻譜具有「共軛對稱性」，所以打點時必須對稱阻擋 (中心點 ± 偏移量)
            cv2.circle(mask, (ccol + dx, crow + dy), block_size, 0, thickness=-1)
            cv2.circle(mask, (ccol - dx, crow - dy), block_size, 0, thickness=-1)

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
    # img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_clean = np.uint8(img_back)               # 轉換資料型態為 uint8

    # 【步驟 6】影像還原與輸出
    # img_back = np.abs(img_back)

    # # === 🌟 升級版：百分位數穩健統計對齊 (Robust Statistical Alignment) ===
    # # 1. 計算原圖的真實亮度與對比度
    # orig_mean = np.mean(img)
    # orig_std = np.std(img)
    
    # # 2. 排除 IFFT 產生的極端偽影數值 (取 2% ~ 98% 的數據範圍)
    # p2, p98 = np.percentile(img_back, (2, 98))
    
    # # 將極端值暫時截斷，避免它們干擾標準差的計算
    # img_back_robust = np.clip(img_back, p2, p98)
    
    # # 3. 使用穩健的平均值與標準差，將影像線性拉伸回原圖的對比度
    # img_aligned = (img_back - np.mean(img_back_robust)) / np.std(img_back_robust) * orig_std + orig_mean
    
    # # 4. 嚴格裁切至 0-255 並轉型
    # img_clean = np.clip(img_aligned, 0, 255).astype(np.uint8)
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

    # # --- Matplotlib 繪圖與視覺化 (支援滑鼠點擊互動) ---
    # fig = plt.figure(figsize=(16, 8))
    
    # plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    # plt.title('1. Original Image'), plt.axis('off')
    
    # plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('2. FFT Spectrum (點擊找星星!)'), plt.axis('off')
    
    # plt.subplot(2, 3, 3), plt.imshow(mask, cmap='gray')
    # plt.title('3. Notch Mask'), plt.axis('off')
    
    # # 注意：這裡使用取絕對值後的頻譜來視覺化
    # plt.subplot(2, 3, 4)
    # magnitude_filtered = 20 * np.log(np.abs(fshift_filtered) + 1e-1)
    # plt.imshow(magnitude_filtered, cmap='gray')
    # plt.title('4. Filtered Spectrum'), plt.axis('off')
    
    # plt.subplot(2, 3, 5), plt.imshow(img_clean, cmap='gray')
    # plt.title('5. Cleaned Result'), plt.axis('off')
    
    # plt.tight_layout()

    # # ==========================================
    # # 🌟 新增：滑鼠點擊偵測系統
    # def onclick(event):
    #     # 確認有點擊到圖表內部，且取得 x, y 座標
    #     if event.xdata is not None and event.ydata is not None:
    #         # Matplotlib 取得的是浮點數，轉成整數
    #         x = int(event.xdata)
    #         y = int(event.ydata)
            
    #         # 計算距離中心的偏移量 dx, dy
    #         dx = x - ccol
    #         dy = y - crow
            
    #         print("\n" + "="*50)
    #         print(f"🖱️ 偵測到點擊！ 原始座標: (x={x}, y={y})")
    #         print(f"🎯 算出的偏移量 (dx, dy): ({dx}, {dy})")
    #         print(f"👉 請將此點加入你的 notch_points 中：")
    #         print(f"   ({dx}, {dy}),")
    #         print("="*50)

    # # 將點擊事件綁定到我們寫的 onclick 函式上
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # # ==========================================

    # plt.show()

    # return img_clean

# ---------------- 使用範例 ----------------
# 處理水平雜訊 (圖 8)
# clean_img_h = remove_periodic_noise('yzu8noiseg.png', noise_type='horizontal', dc_radius=20, block_size=3)
# cv2.imwrite("yzu8noiseg_denoised_2.png", clean_img_h)

# 處理網格雜訊 (圖 7)
# 假設你觀察到亮點在距離中心 (x=50, y=30) 以及 (x=50, y=-30) 的位置
clean_img_g = remove_periodic_noise('yzu7noiseg.png', noise_type='grid', dc_radius=50, h_blocksize=1, v_blocksize=1)
cv2.imwrite("yzu7noiseg_denoised_2.png", clean_img_g)