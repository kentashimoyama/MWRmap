import csv
import os


# input_filename = "20240717_1721193682_adv_111_right_utc.csv"
# outputdirname = "\csv_0717_agawall_withoutlier_1_right"


input_filename_list = ["240125_144504261_dat_utc_right_2.csv", "240125_144508544_dat_utc_left_2.csv"]
outputdirname_list = ["\csv_vison_right", "\csv_vison_left"]

for idx, input_filename in enumerate(input_filename_list):
    print(f"process {input_filename} ...")

    #input_file = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/240125_151137862_dat_amp80_utc.csv"
    # input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco/generate_mvp/mwr/"+input_filename
    input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr/"+input_filename

    with open(input_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        header = next(reader)  # ヘッダー行を読み飛ばす

        # 8列目のインデックス
        target_column_index = 7

        # 14列目の数字ごとにCSVを分割するためのディレクトリを作成
        #output_directory = "C:\workspace\MELCO\data/20240123_VISON\MWR/240125_151137862_vison012506r/csv80"
        # output_directory = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr"+outputdirname_list[idx]
        output_directory = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr/"+outputdirname_list[idx]
        os.makedirs(output_directory, exist_ok=True)

        # 分割されたCSVファイルの書き込みハンドルを格納するディクショナリ
        output_files = {}
        i = 0
        for row in reader:
            print(i)
            target_value = row[target_column_index]

            # 対応するファイルハンドルを取得するか、存在しない場合は新規に作成する
            if target_value not in output_files:
                output_file = os.path.join(output_directory, f"{target_value}.csv")
                output_files[target_value] = open(output_file, 'w', newline='')
                writer = csv.writer(output_files[target_value])
                #writer.writerow(header)  # ヘッダー行を書き込む
            else:
                writer = csv.writer(output_files[target_value])

            writer.writerow(row)  # 行を書き込む

            i+=1

        # ファイルハンドルを閉じる
        for file in output_files.values():
            file.close()