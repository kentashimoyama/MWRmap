import csv
import datetime
import math
import os
from config import Set_config

class Mwrunix2utc():
    def __init__(self):
        self.a = None
        sc = Set_config()
        self.home_dir = sc.home_dir
    
    def convert_unix_to_utc(self, unix_time):
        if unix_time == 'UNIXTIME':
            return unix_time  # 'UNIXTIME' の場合はそのまま返す

        # utc_time = datetime.datetime.utcfromtimestamp(float(unix_time))
        utc_time = datetime.datetime.fromtimestamp(float(unix_time), datetime.UTC)
        # utc_time += datetime.timedelta(hours=9)  # UTC時間から9時間足す
        return utc_time.strftime("%H%M%S.%f")

    def main(self, file_name, whitelane_time):
        input_file = self.home_dir+"\mwr/"+file_name
        # input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr/"+file_name
        output_file = input_file+"_utc.csv"

        if os.path.exists(output_file):
            os.remove(output_file)  # ファイルを削除
            print(f"{output_file} は削除されました。")
        else:
            print(f"{output_file} は存在しません。")

        with open(input_file+".csv", 'r', encoding='utf-8') as csv_input, open(output_file, 'w', newline='') as csv_output:
            reader = csv.reader(csv_input)
            writer = csv.writer(csv_output)

            next(reader)  # 最初の行をスキップ

            for row in reader:
                unix_time = row[2]  # 11列目のデータを取得
                if whitelane_time > float(unix_time):
                    continue
                # print(unix_time)
                # print(type(unix_time))
                utc_time = self.convert_unix_to_utc(unix_time)  # UNIX時間をUTC時間に変換

                # value_4 = float(row[3])  # 4行目の値を取得
                # value_6 = float(row[5])  # 6行目の値を取得

                # value_15 = value_4 * math.sin(math.radians(value_6))
                # value_16 = value_4 * math.cos(math.radians(value_6))

                row.append(utc_time)  # 14列目にUTC時間を追加
                # row.append(value_15)  # 15列目に計算結果を追加
                # row.append(value_16)  # 16列目に計算結果を追加
                writer.writerow(row)


if __name__ == "__main__":
    mwrunix2utc = Mwrunix2utc()
    # file_name_list = ["20240907_1725699643_adv_1right", "20240907_1725699657_adv_1left", "20240907_1725700271_adv_2right", "20240907_1725700273_adv_2left", "20240907_1725700578_adv_3right", "20240907_1725700579_adv_3left"]
    file_name_list = ["20241231_1735635244_adv_2rightwall0", "20241231_1735635247_adv_2leftwall0"]

    whitelane_time_list = [0, 0]

    for i, file_name in enumerate(file_name_list):
        #file_name = "20240717_1721193906_adv_222_right_80azi_cosinlier"

        print(f"processing mwr unix to utc ... input filename : {file_name}")

        mwrunix2utc.main(file_name, whitelane_time_list[i])

        # input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr/"+file_name
        # output_file = input_file+"_utc"

        # with open(input_file+".csv", 'r') as csv_input, open(output_file+".csv", 'w', newline='') as csv_output:
        #     reader = csv.reader(csv_input)
        #     writer = csv.writer(csv_output)

        #     next(reader)  # 最初の行をスキップ

        #     for row in reader:
        #         unix_time = row[2]  # 11列目のデータを取得
        #         # print(unix_time)
        #         # print(type(unix_time))
        #         utc_time = convert_unix_to_utc(unix_time)  # UNIX時間をUTC時間に変換

        #         # value_4 = float(row[3])  # 4行目の値を取得
        #         # value_6 = float(row[5])  # 6行目の値を取得

        #         # value_15 = value_4 * math.sin(math.radians(value_6))
        #         # value_16 = value_4 * math.cos(math.radians(value_6))

        #         row.append(utc_time)  # 14列目にUTC時間を追加
        #         # row.append(value_15)  # 15列目に計算結果を追加
        #         # row.append(value_16)  # 16列目に計算結果を追加
        #         writer.writerow(row)
