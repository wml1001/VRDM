import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd

ssim_recode = []
for index in range(1,703):
    frog = cv.imread(f"....",cv.IMREAD_GRAYSCALE)
    no_frog = cv.imread(f"...",cv.IMREAD_GRAYSCALE)

    ssim_recode.append(ssim(frog,no_frog))

print(np.mean(ssim_recode))

df = pd.DataFrame(ssim_recode, columns=['ssim value:'])

# 将 DataFrame 写入 Excel 文件
df.to_excel('bbdm_ssim.xlsx', index=False)
