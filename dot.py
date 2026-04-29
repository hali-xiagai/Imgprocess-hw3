import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像並計算頻譜
# ⚠️ 請確保這裡的檔名是你目前的雜訊圖
image_path = 'yzu7noiseg.png' 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ 無法讀取影像，請確認圖片是否存在於同一資料夾！")
    exit()

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # 計算中心點

# 進行傅立葉轉換取得頻譜
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-1)

# 2. 準備儲存座標的清單
collected_points = []

# 點擊事件處理函式
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # 取得點擊座標並計算與中心的偏移量 dx, dy
        x = int(event.xdata)
        y = int(event.ydata)
        dx = x - ccol
        dy = y - crow
        
        # 將偏移量存入清單，並印出一行提示
        collected_points.append((dx, dy))
        print(f"✅ 已記錄點擊: (dx={dx}, dy={dy})")

# 3. 顯示圖片並綁定點擊事件
fig, ax = plt.subplots(figsize=(10, 10))

# 為了讓你更容易看清楚星星，我們稍微調整一下顯示的對比度 (vmax)
vmax = np.percentile(magnitude_spectrum, 99.5) 
ax.imshow(magnitude_spectrum, cmap='gray', vmax=vmax)
ax.set_title(f"FFT Spectrum\nCenter DC is at ({ccol}, {crow})\nClick on the bright noise stars! Close window to finish.")
ax.axis('off')

# 綁定滑鼠點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)

print("=" * 50)
print("🖱️ 採集工具已啟動！")
print("1. 請在彈出的視窗中，點擊那些異常明亮的雜訊星星。")
print("2. 點擊完畢後，請直接「關閉視窗 (按打叉)」。")
print("=" * 50)

# 程式會停在這裡，直到你關閉 Matplotlib 視窗
plt.show()

# 4. 視窗關閉後，一口氣輸出排版好的最終結果
print("\n" + "=" * 50)
print("🎉 座標收集完成！請將以下程式碼直接複製貼上到 gemini.py 中：\n")

print("notch_points = [")
for pt in collected_points:
    print(f"    {pt},")
print("]")

print("=" * 50)