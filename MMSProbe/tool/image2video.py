import cv2
import glob
from tqdm import tqdm

folder = r'E:\workspace\MMSProbe_2022\data\MWR_results\vison\20221212221053'
img_array = []
files = glob.glob(folder + '/*.png')
files.sort()
for filename in tqdm(files):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(folder + '/video.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()