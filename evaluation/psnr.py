import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import pandas as pd

psnr_recode = []
for index in range(1,703):
    frog = cv.imread(f"...",cv.IMREAD_GRAYSCALE)
    no_frog = cv.imread(f".....",cv.IMREAD_GRAYSCALE)

    psnr_recode.append(psnr(frog,no_frog))

print(np.mean(psnr_recode))

df = pd.DataFrame(psnr_recode, columns=['psnr value:'])

# 将 DataFrame 写入 Excel 文件
df.to_excel('bbdm_psnr.xlsx', index=False)
