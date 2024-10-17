import csv
import datetime
import math
import pandas as pd
import matplotlib as plt
import numpy as np

def convert_unix_to_utc(unix_time):
    if unix_time == 'UNIXTIME':
        return unix_time  # 'UNIXTIME' の場合はそのまま返す

    utc_time = datetime.datetime.utcfromtimestamp(float(unix_time))
    utc_time -= datetime.timedelta(hours=9)  # UTC時間から9時間引く
    return utc_time.strftime("%H%M%S.%f")


#amplitude閾値
def write_amp_hist(file): #6mwr1 th:>80に設定
    df = pd.read_csv(file)
    amp = df["Amplitude"]
    plt.hist(amp, bins=300, edgecolor="black")
    plt.title("Histgram of Amplitude")
    plt.xlabel("Amplitude dB")
    plt.ylabel("Frequency")
    plt.xlim(left=40, right=130)
    plt.savefig(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr\hist.png")
    plt.close()

def amp_th(file, th):
    index_list = []
    df = pd.read_csv(file)
    amp = df["amplitude"]
    amp_list = list(amp)
    for i ,j in enumerate(amp_list):
        print(i)
        if j > th:
            index_list.append(i)
    inlier_df = df.iloc[index_list]

    return inlier_df

def azimath_th(file):
    index_list = []
    df = pd.read_csv(file) 
    azimuth = df["Azimuth[deg]"].to_numpy()
    # x = df['Xrange'].to_numpy()
    # y = df['Yrange'].replace(0, 1e-9).to_numpy()
    # angle = np.degrees(np.arctan(x / y))
    for i, j in enumerate(azimuth):
        print(i)
        if abs(j) < 30:
            index_list.append(i)
    inlier_df = df.iloc[index_list]
    
    return inlier_df

def amp_azimath_th(file, th):
    index_list = []
    df = pd.read_csv(file)
    # df.columns = ["X_cor", "Y_cor", "Amplitude", "UnixTime", "Xrange", "Yrange", "Velocity[m/s]"]
    amp = df["amplitude"]
    amp_list = list(amp)
    for i ,j in enumerate(amp_list):
        print(f"processing amp ... {i+1}")
        if j > th:
            index_list.append(i)
    inlier_amp_df = df.iloc[index_list]

    index_list = []
    azimuth = inlier_amp_df["Azimuth[deg]"].to_numpy()
    # x = inlier_amp_df['Xrange'].to_numpy()
    # y = inlier_amp_df['Yrange'].replace(0, 1e-9).to_numpy()
    # angle = np.degrees(np.arctan(x / y))
    for i, j in enumerate(azimuth):
        print(f"processing azi ... {i}")
        if abs(j) < 30:
            index_list.append(i)
    inlier_ampazi_df = inlier_amp_df.iloc[index_list]

    return inlier_ampazi_df


ampth = 80

# input_file = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/240125_151137862_dat.csv"
# output_file = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/240125_151137862_dat_utc.csv"
# input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144504261_vison012503r/240125_144504261_dat.csv"
input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144508544_vison012503l/240125_144508544_dat.csv"
# output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144504261_vison012503r/240125_144504261_dat_utc.csv"
# output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144504261_vison012503r/240125_144504261_dat_amp"+str(ampth)+"_utc.csv"
output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144508544_vison012503l/240125_144508544_dat_utc.csv"
# output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144508544_vison012503l/240125_144508544_dat_amp"+str(ampth)+"_utc.csv"

# inlier_df = amp_th(input_file, ampth)
# inlier_df = azimath_th(input_file)
# inlier_df = amp_azimath_th(input_file, ampth)

# reader = list(np.array(inlier_df))

with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
    reader = csv.reader(csv_input)
    writer = csv.writer(csv_output)

    next(reader)  # 最初の行をスキップ

    for row in reader:
        # row = list(row)
        unix_time = row[10]  # 11列目のデータを取得
        utc_time = convert_unix_to_utc(unix_time)  # UNIX時間をUTC時間に変換

        value_4 = float(row[3])  # 4行目の値を取得
        value_6 = float(row[5])  # 6行目の値を取得

        value_15 = value_4 * math.sin(math.radians(value_6))
        value_16 = value_4 * math.cos(math.radians(value_6))

        row.append(utc_time)  # 14列目にUTC時間を追加
        row.append(value_15)  # 15列目に計算結果を追加
        row.append(value_16)  # 16列目に計算結果を追加
        writer.writerow(row)
