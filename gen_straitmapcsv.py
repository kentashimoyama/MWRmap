import csv
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd
import re
from config import Set_config


class Gen_straightmap:
    def __init__(self,dir_name, start_idnum):
        sc = Set_config()
        home_dir = sc.home_dir
        # ディレクトリのパスを指定
        self.directory_path = home_dir + f'/wall/{dir_name}/'  # ここを適切なパスに変更してください
        # IDの初期値
        self.id_counter = start_idnum

    # 数字を抽出して昇順にソートする関数
    def extract_number(self, filename):
        match = re.search(r'\d+', filename)  # ファイル名から数字を抽出
        return int(match.group()) if match else float('inf')  # 数字が見つからない場合は無限大を返す

    def main(self):
        output_file = 'output.csv'
        # 新しいデータを格納するリスト
        output_data = []
        
        # ファイル名を数字の昇順にソート
        file_list = os.listdir(self.directory_path)
        sorted_file_list = sorted(file_list, key=self.extract_number)


        # ディレクトリ内のすべてのCSVファイルを処理
        for filename in sorted_file_list:
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory_path, filename)
                # CSVファイルを読み込む
                print(file_path)
                # exit()
                df = pd.read_csv(file_path, header=None)

                # 1行目とn行目のデータを取得
                first_row = df.iloc[0]
                last_row = df.iloc[-1]

                # 新しいデータをリストに追加 (idは20刻みで増加)
                output_data.append([first_row[1], first_row[0], None, None, self.id_counter])
                output_data.append([last_row[1], last_row[0], None, None,  self.id_counter])
                self.id_counter += 20  # IDを20刻みで増加

        # 新しいCSVファイルに保存
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(self.directory_path+output_file, index=False, header=False)

        print(f'{output_file} にデータを保存しました。')


if __name__ == "__main__":
    dir_name = 'symbols_0120_3rightwall3/rightwall'
    start_idnum = 480 #mwrベクター地図上の直線につけるidの初期値

    gs = Gen_straightmap(dir_name, start_idnum)
    gs.main()




# # ディレクトリのパスを指定
# directory_path = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\wall\symbols_1231_2rightwall0_kikuichover2_15\rightwall/'  # ここを適切なパスに変更してください
# output_file = 'output.csv'

# # 新しいデータを格納するリスト
# output_data = []

# # IDの初期値
# id_counter = 100




# # 数字を抽出して昇順にソートする関数
# def extract_number(filename):
#     match = re.search(r'\d+', filename)  # ファイル名から数字を抽出
#     return int(match.group()) if match else float('inf')  # 数字が見つからない場合は無限大を返す

# # ファイル名を数字の昇順にソート
# file_list = os.listdir(directory_path)
# sorted_file_list = sorted(file_list, key=extract_number)


# # ディレクトリ内のすべてのCSVファイルを処理
# for filename in sorted_file_list:
#     if filename.endswith('.csv'):
#         file_path = os.path.join(directory_path, filename)
#         # CSVファイルを読み込む
#         print(file_path)
#         # exit()
#         df = pd.read_csv(file_path, header=None)

#         # 1行目とn行目のデータを取得
#         first_row = df.iloc[0]
#         last_row = df.iloc[-1]

#         # 新しいデータをリストに追加 (idは20刻みで増加)
#         output_data.append([first_row[1], first_row[0], None, None, id_counter])
#         output_data.append([last_row[1], last_row[0], None, None,  id_counter])
#         id_counter += 20  # IDを20刻みで増加

# # 新しいCSVファイルに保存
# output_df = pd.DataFrame(output_data)
# output_df.to_csv(directory_path+output_file, index=False, header=False)

# print(f'{output_file} にデータを保存しました。')

