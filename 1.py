import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# 你的现有代码
ssim_value = []

for index in range(1, 5627):
    noise_matrix = np.random.randn(64, 64)
    img = Image.open(f'/home/back_door/data7000/data/train/A/{index}.png').convert('L')
    img_array = np.array(img)
    img_array_with_noise = img_array + noise_matrix
    ssim_value.append(int(ssim(img_array, img_array_with_noise, data_range=255, multichannel=False)*10000000))
# for index in range(1, 703):
#     frog = Image.open(f'C:/Users/31435/Desktop/test/test/test/A/frog/{index}.png').convert('L')
#     no_frog = Image.open(f'C:/Users/31435/Desktop/test/test/test/B/no_frog/{index}.png').convert('L')
#     img_array1 = np.array(frog)
#     img_array2 = np.array(no_frog)
#     ssim_value.append(ssim(img_array1, img_array2, data_range=255, multichannel=False))


tem1 = np.min(ssim_value)
tem2 = np.max(ssim_value)
print(tem1,"===",tem2)
# for index,value in enumerate(ssim_value):
#     temp = ((value  - tem1)/(tem2 - tem1))
#     # print(value)
#     ssim_value[index] = int(100 + temp*100)


# # 使用pandas创建DataFrame
# df = pd.DataFrame(ssim_value, columns=['SSIM_Value'])

# # 将DataFrame保存到Excel文件中
# excel_filename = 'SSIM_Values.xlsx'  # Excel文件名
# df.to_excel(excel_filename, index=False)  # 保存，不包含行索引

# print(f'SSIM values have been saved to {excel_filename}')