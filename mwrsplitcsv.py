import csv
import os
import pandas as pd
import shutil
from config import Set_config

class MwrSplit():
    def __init__(self):
        self.a = None
        sc = Set_config()
        self.home_dir = sc.home_dir

    def main(self, input_filename, output_dir):
        input_file = self.home_dir + "/mwr/"+input_filename
        with open(input_file, 'r', encoding='cp932', errors='ignore') as csv_input: #csvファイル読み込み
            # with pd.ExcelFile(input_file) as xls: #エクセルファイル読み込み
            #     df = pd.read_excel(xls, sheet_name='Sheet1')  # シート名を指定
            
            reader = csv.reader(csv_input)
            header = next(reader)  # ヘッダー行を読み飛ばす

            # 8列目のインデックス
            target_column_index = 7

            # 14列目の数字ごとにCSVを分割するためのディレクトリを作成
            #output_directory = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/csv80"
            # output_directory = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr/"+output_dir
            output_directory = self.home_dir + "/mwr/"+output_dir
            if os.path.exists(output_directory) and os.path.isdir(output_directory):
                shutil.rmtree(output_directory)  # ディレクトリを削除
                print(f"{output_directory} は削除されました。")
            else:
                print(f"{output_directory} は存在しません。")
            os.makedirs(output_directory, exist_ok=True)

            # 分割されたCSVファイルの書き込みハンドルを格納するディクショナリ
            output_files = {}
            i = 0

            for row in reader:
                target_value = row[target_column_index]
                # print(row)

                # 対応するファイルハンドルを取得するか、新規に作成する
                if target_value not in output_files:
                    output_file = os.path.join(output_directory, f"{target_value}.csv")
                    
                    # with文を使用してファイルを書き込む
                    with open(output_file, 'w', newline='') as outfile:
                        output_files[target_value] = outfile
                        writer = csv.writer(outfile)
                        # writer.writerow(header)  # ヘッダー行を書き込む
                        writer.writerow(row)  # 行を書き込む
                        # print(f"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:{i}")
                        # output_file = os.path.join(output_directory, f"{target_value}.csv")
                       
                        i += 1
                else:
                    # 既存のファイルに書き込む
                    with open(os.path.join(output_directory, f"{target_value}.csv"), 'a', newline='') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(row)  # 行を書き込む
                        # print(f"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:{i}")
                        i += 1
                
            # exit()

            # for row in reader:
            #     # print(i)
            #     target_value = row[target_column_index]

            #     # 対応するファイルハンドルを取得するか、存在しない場合は新規に作成する
            #     if target_value not in output_files:
            #         output_file = os.path.join(output_directory, f"{target_value}.csv")
            #         output_files[target_value] = open(output_file, 'w', newline='')
            #         writer = csv.writer(output_files[target_value])
            #         #writer.writerow(header)  # ヘッダー行を書き込む
            #     else:
            #         writer = csv.writer(output_files[target_value])

            #     writer.writerow(row)  # 行を書き込む

            #     i+=1

            # # ファイルハンドルを閉じる
            # for file in output_files.values():
            #     file.close()
        

# input_filename = "20240717_1721193682_adv_111_right_utc.csv"
# outputdirname = "\csv_0717_agawall_withoutlier_1_right"

if __name__ == "__main__":
    ms = MwrSplit()
    input_filename_list = ["20240125_144508544_dat_3left_utc.csv", "20240125_144504261_dat_3right_utc.csv"] #mwrデータ、x_range,t_range,unixtimeが入ってる列が違うかもだから確認してみて
    outputdirname_list = ["csvall_0125_3leftwall1", "csvall_0125_3rightwall1"] #出力ディレクトリ

    for idx, input_filename in enumerate(input_filename_list):
        print(f"processing mwr split ... input filename : {input_filename}")

        #input_file = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/240125_151137862_dat_amp80_utc.csv"
        # input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco/generate_mvp/mwr/"+input_filename
        # input_file = ms.home_dir + "/mwr/"+input_filename
        output_dir =  outputdirname_list[idx]

        ms.main(input_filename, output_dir)

        # with open(input_file, 'r') as csv_input:
        #     reader = csv.reader(csv_input)
        #     header = next(reader)  # ヘッダー行を読み飛ばす

        #     # 8列目のインデックス
        #     target_column_index = 7

        #     # 14列目の数字ごとにCSVを分割するためのディレクトリを作成
        #     #output_directory = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/csv80"
        #     output_directory = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr"+outputdirname_list[idx]
        #     os.makedirs(output_directory, exist_ok=True)

        #     # 分割されたCSVファイルの書き込みハンドルを格納するディクショナリ
        #     output_files = {}
        #     i = 0
        #     for row in reader:
        #         print(i)
        #         target_value = row[target_column_index]

        #         # 対応するファイルハンドルを取得するか、存在しない場合は新規に作成する
        #         if target_value not in output_files:
        #             output_file = os.path.join(output_directory, f"{target_value}.csv")
        #             output_files[target_value] = open(output_file, 'w', newline='')
        #             writer = csv.writer(output_files[target_value])
        #             #writer.writerow(header)  # ヘッダー行を書き込む
        #         else:
        #             writer = csv.writer(output_files[target_value])

        #         writer.writerow(row)  # 行を書き込む

        #         i+=1

        #     # ファイルハンドルを閉じる
        #     for file in output_files.values():
        #         file.close()