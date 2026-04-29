import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def remove_periodic_noise(image_path, noise_type='horizontal', dc_radius=20, block_size=20, h_blocksize=5, v_blocksize=5, h=10):
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
    elif noise_type == 'cross':
    # 步驟一：先做所有「破壞 (設為 0)」的動作
        mask[:, ccol-v_blocksize : ccol+v_blocksize+1] = 0  # 畫垂直黑條
        mask[crow-h_blocksize : crow+h_blocksize+1, :] = 0  # 畫水平黑條
        
        # 步驟二：最後做「保護 (設為 1)」的動作，確保中心絕對不會被剛剛的黑線蓋住
        cv2.circle(mask, (ccol, crow), dc_radius, 1, thickness=-1)

    # 【步驟 4】頻域濾波 (Element-wise multiplication)
    fshift_filtered = fshift * mask

    # 供視覺化觀察濾波後的頻譜
    magnitude_filtered = 20 * np.log(np.abs(fshift_filtered) + 1e-1)

    # 【步驟 5】頻率域轉換回空間域 (IFFT)
    f_ishift = np.fft.ifftshift(fshift_filtered) # 將中心點移回原點
    img_back = np.fft.ifft2(f_ishift)            # 二維反傅立葉轉換

    # 【步驟 6】影像還原與輸出
    img_back = np.abs(img_back)                  # 取絕對值 (Magnitude)
    brightness_shift = np.mean(img) - np.mean(img_back)
    img_back = img_back + brightness_shift
    img_clean = np.uint8(img_back)               # 轉換資料型態為 uint8
    img_clean = cv2.fastNlMeansDenoising(img_clean, None, h, templateWindowSize=7, searchWindowSize=21)

    return img_clean

# ---------------- 使用範例 ----------------
# 處理水平雜訊 (圖 8)
# clean_img = remove_periodic_noise('yzu8noiseg.png', noise_type='horizontal', dc_radius=20, block_size=4, h=2)
# cv2.imwrite("yzu8noiseg_denoised.png", clean_img)

# 處理網格雜訊 (圖 7)
# 假設你觀察到亮點在距離中心 (x=50, y=30) 以及 (x=50, y=-30) 的位置
# clean_img = remove_periodic_noise('yzu7noiseg.png', noise_type='cross', dc_radius=20, h_blocksize=2, v_blocksize=2, h=10)
# cv2.imwrite("yzu7noiseg_denoised.png", clean_img)
if __name__ == "__main__":
    paraser = argparse.ArgumentParser(description='Remove periodic noise from an image using a notch filter.')
    paraser.add_argument('--image_path', type=str, required=True, help='Path to the input image with periodic noise.')

    args = paraser.parse_args()
    image_path = args.image_path
    image_name = image_path.split('/')[-1]
    if 'yzu8noiseg' in image_name:
        clean_img = remove_periodic_noise(image_path, noise_type='horizontal', dc_radius=20, block_size=4, h=2)
        cv2.imwrite(f"denoised_{image_name}", clean_img)
    elif 'yzu7noiseg' in image_name:
        clean_img = remove_periodic_noise(image_path, noise_type='cross', dc_radius=20, h_blocksize=2, v_blocksize=2, h=10)
        cv2.imwrite(f"denoised_{image_name}", clean_img)