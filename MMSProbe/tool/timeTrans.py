import glob
import os
import cv2
import csv
from datetime import datetime, timezone

TIME_FORMAT = "%y%m%d%H%M%S%f"
TIME_ZONE_INFO = timezone.utc

def timestamp2str(t: float):
	dt = datetime.fromtimestamp(t, TIME_ZONE_INFO)
	return dt.strftime(TIME_FORMAT)

folder = r'E:\workspace\img_GPS\2201191617'
files = glob.glob(r'I:\20220119\data\debug_202201191617\Main\image_out\*.jpg')
time_stamps = []
for file in files:
    filename = os.path.basename(file).split('.')
    time_stamp = timestamp2str(float(filename[0]+'.'+filename[1]))
    cv2.imwrite(os.path.join(folder, time_stamp[6:15] + '.jpg'), cv2.imread(file))
    time_stamps.append(time_stamp)

lats = []
lons = []

path = r'I:\20220119\data\debug_202201191617/Maingps_raw_latlon.csv'
with open(path) as f:
    reader = csv.reader(f)
    for row in reader:
        lats.append(row[0])
        lons.append(row[1])

output_path = r'E:\workspace\img_GPS\2201191617\GPSData.csv'
with open(output_path, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(len(time_stamps)):
        row = [time_stamps[i][6:15], lats[i], lons[i]]
        print(row)
        writer.writerow(row)