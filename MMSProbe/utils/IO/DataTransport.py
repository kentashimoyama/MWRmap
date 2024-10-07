import zmq
import time
import pandas as pd
import numpy as np
import glob
import os
import struct
from datetime import datetime, timezone

TIME_FORMAT = "%y%m%d%H%M%S%f"
TIME_ZONE_INFO = timezone.utc
MMSP_MWR_OFFSET = 9*60*60*1000
#MMSP_MWR_OFFSET = 9*60*60*1000 + 4*1000
#MMSP_MWR_OFFSET = 241 * 1000

def str2timestamp(time_str: str):
	t = datetime.strptime(time_str, TIME_FORMAT)
	t = t.replace(tzinfo = TIME_ZONE_INFO)
	return round(t.timestamp(), 6)

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5000")

# root = 'E:\MELCO\data\work/'
root = r'C:\workspace\MELCO\data\20240123_VISON\MMSProbe/'
folder = '240125055827'

# GPSデータを読み込む
GPSData = dict()
fr = open(root + folder + "/GPSData.csv", "r")
for line in fr:
    row = line[:-1].split(",")
    GPSData[row[0]] = np.asarray(row[1:4], float)
fr.close()

# ミリ波レーダーシンボルを読み込む
MWRData = dict()
# test1-2:220217060303
# test2-1:220217060807

#df_milliwave = pd.read_csv(r'C:\workspace\MELCO\data\20231122_KIKUICHO\MWR\231122_163106919_TEST_006_R/231122_163106919_dat_utc_over70.csv',sep=',', encoding='utf-8')
#df_milliwave = df_milliwave[['UnixTime', 'M_Xrange[m]', 'M_Yrange[m]', 'ID']]

#変更
file_path = r'C:\workspace\MELCO\data\20240123_VISON\MWR\240125_145828606_vison012504r/240125_145828606_dat_amp65_utc.csv'
df_milliwave = pd.read_csv(file_path, header=None, usecols=[1, 10, 14, 15],sep=',', encoding='utf-8')
df_milliwave.columns = ['ID', 'UnixTime', 'M_Xrange[m]', 'M_Yrange[m]']

MWR_time_list = sorted(list(set(df_milliwave['UnixTime'])))
MWR_dict = {}
for MWR_time in MWR_time_list:
    MWR_dict[MWR_time] = df_milliwave[df_milliwave['UnixTime'] == MWR_time]

images = glob.glob(root + folder+'/*.raw')
row = 720
col = 1280
type = 1
info = [row, col, type]
images.sort()

i_flag = 0
for frameNumber, image in enumerate(images):
    send_flag = 0
    time_id = os.path.basename(image).split(".")[0]
    raw = np.fromfile(image, dtype = "uint8")
    jpg = raw.reshape((720,1280,3))
    # jpg = cv2.imread(image)
    for i in range(3):
        num = info[i]
        val = struct.pack('i', num)
        publisher.send(val, zmq.SNDMORE)

    t = int(str2timestamp(folder[:6] + time_id)*1000)
    msg = struct.pack('Q', t)
    publisher.send(msg, zmq.SNDMORE)
    for i in range(3):
        msg = struct.pack('d', GPSData[time_id][i])
        publisher.send(msg, zmq.SNDMORE)
    publisher.send(struct.pack('Q', frameNumber), zmq.SNDMORE)
    publisher.send(struct.pack('d', time.time()), zmq.SNDMORE)
    if  frameNumber + 1 == len(images):
        publisher.send(struct.pack('i', 1), zmq.SNDMORE)
    else:
        publisher.send(struct.pack('i', 0), zmq.SNDMORE)
    for i, MWR_time in enumerate(np.array(MWR_time_list[i_flag:])*1000):
        if MWR_time- MMSP_MWR_OFFSET<= t < MWR_time_list[i_flag + i + 1]*1000 - MMSP_MWR_OFFSET:

            i_flag += i
            send_flag = 1
            break
    publisher.send(struct.pack('i', send_flag))
    if send_flag:
        publisher.send_pyobj(MWR_dict[MWR_time/1000], zmq.SNDMORE)
        print(MWR_dict[MWR_time/1000])
    publisher.send_pyobj(jpg)

    print(frameNumber)
    print(t)
    time.sleep(0.3)