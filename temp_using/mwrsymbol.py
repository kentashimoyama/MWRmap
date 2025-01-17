#TODO
#ver3から変換を変更.
"""
IWR_DIR：mwrの出力ファイル _adv.csvをmwrsplitcsv.pyを使ってフレームIDごとに別のcsvファイルに分割して出力した結果を格納したDIR
設定されたamplitudeを閾値にして切ったものも用意
AQLOC_FILE:aqlocの出力ファイル .log を.csvに直し, gpggato19.pyで19系座標に直したもの
変更するべき箇所
gpggato19.pyの測地系の変換を9系に設定．
gpggato19.pyにyaw角計算関数を追加したが不正確．

mwrのqx, qyを
row[14] -> row[3]
row[15] -> row[4]

straightgeneration.py の出力が傾き，切片だけしかない．x座標の最小値と最大値は? inlierの最小値と最大値．壁を映したと思われるフレームすべてに対して直線近似を行う．　1フレームごとに直線近似してるところはどこ???
→trim平均で10%ずつ(計20%)y切片除去，傾きの外れ値を20%除外．つまり使うフレームは60%のみ(y切片で20%, 傾きで20%除去)
→そのすべての直線の傾きと切片の平均を取る．x座標の最小値と最大値はinlierの最小値と最大値
"""


import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np

iwrdirname_list = ["csv_vison_right", "csv_vison_left"]
aqlocfilename_list = ["012503_latlon19.csv", "012503_latlon19.csv"]
finalfilename_list = ["symbols_vison_right.csv", "symbols_vison_left.csv"]

right_list = [True, False]



for iwrdirname, aqlocfilename, finalfilename, right in zip(iwrdirname_list, aqlocfilename_list, finalfilename_list, right_list):
    # filepath
    # IWR_DIR = r"C:\workspace\MELCO\data\melcomwr\230605_194737459_008\split_csv_coscurvenoisereduct"
    # AQLOC_FILE = r"C:\workspace\MELCO\data\AQLOC\060508out19.csv"

    # IWR_DIR = r"C:\workspace\MELCO\data/20231227_KIKUICHO\MWR/231228_155842518_TEST_005/1"
    # AQLOC_FILE = r"C:\workspace\MELCO\data\20231227_KIKUICHO\AQLOC/122705_ichimill19hokan.csv"

    # IWR_DIR = r"C:\workspace\MELCO\data/20231020_KTF/03/frame2"
    # AQLOC_FILE = r"C:\workspace\MELCO\data\20231020_KTF/nmea_GPGGAlatlon19.csv"

    #IWR_DIR = r"C:\workspace\MELCO\data\20240123_VISON\MWR\240125_144504261_vison012503r\csv80"
    #AQLOC_FILE = r"C:\workspace/MELCO\data\20240123_VISON\AQLOC/012503_19.csv"

    IWR_DIR = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr/"+iwrdirname
    AQLOC_FILE = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\AQLOC/'+aqlocfilename

    IWR_TIMESTAMP = []
    data_list = []
    data_listt = []

    # ディレクトリ内のCSVファイルを順に読込
    for filename in os.listdir(IWR_DIR):

        if filename.endswith(".csv"): # CSVファイルのみ処理する
            filepath = os.path.join(IWR_DIR, filename)
            # ファイル名(タイムスタンプ)を拡張子無しで追加
            IWR_TIMESTAMP = float(os.path.splitext(os.path.basename(filename))[0])

            #IWRのタイムスタンプに近いAQLOCの行を探索
            min_diff = float("inf")
            closest_value = None

            with open(AQLOC_FILE, "r", encoding="utf-8_sig") as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if len(row) >= 1:
                        AQLOC_TIME = float(row[0])
                        diff = abs(IWR_TIMESTAMP - AQLOC_TIME)
                        if diff < min_diff:
                            min_diff = diff
                            time = AQLOC_TIME
                            timerow = row

            # if timerow[3] != '#DIV/0!':
            if len(timerow) != 3:
                if timerow[3] == "":
                    continue

                #AQLOC1frameでの位置姿勢
                # px = float(timerow[1])
                # py = float(timerow[2])
                px = float(timerow[2])
                py = float(timerow[1])

                theta = float(timerow[3])

                #print(px, py, theta)

                #IWR1frameのシンボルx,yを格納
                with open(filepath, "r", encoding="utf-8") as csvfile:
                    csvreader = csv.reader(csvfile)
                    header = next(csvreader)
                    for row in csvreader:
                        print(f"processing ... iwrfilename : {iwrdirname} , frame : {time}")

                        # IWR1frameでの位置
                        # qx = float(row[14])
                        # qy = float(row[15])

                        qx = float(row[3])
                        qy = float(row[4])

                        #正面向いている場合
                        # sin_theta = math.cos(theta)
                        # cos_theta = math.sin(theta)
                        # 斜め90deg取付の場合
                        # sin_theta = math.cos(theta + math.pi / 2)
                        # cos_theta = math.sin(theta + math.pi / 2)                    

                        #IWRデータ座標変換

                        

                        data_listt.append([px, py])
                        # iwrx = px - (qx+0.05) * cos_theta + (qy+0.35) * sin_theta #調整をかける x方向(縦断)350mm, y方向(横断)50mmの車両モデル, 20230808の資料参照．px:aqloc, qx:mwr
                        # iwry = py + (qx+0.05) * sin_theta + (qy+0.35) * cos_theta #調整をかける

                        #斜め45deg取付の場合 右
                        if right:
                            # sin_theta = math.cos(theta + math.pi / 4) 
                            # cos_theta = math.sin(theta + math.pi / 4)
                            # sin_theta = math.sin(theta + math.pi / 4) 
                            # cos_theta = math.cos(theta + math.pi / 4)
                            sin_theta = math.sin(-theta - math.pi / 4) 
                            cos_theta = math.cos(-theta - math.pi / 4)
                            sin2_theta = math.sin(-theta) 
                            cos2_theta = math.cos(-theta)
                            # iwrx = px + (qx+0.31) * cos_theta - (qy+0.09) * sin_theta #調整をかける、right
                            # iwry = py + (qx+0.31) * sin_theta + (qy+0.09) * cos_theta #調整をかける, right
                            iwrx = px + cos_theta*qx - sin_theta*qy + cos2_theta*0.31 - sin2_theta*0.09
                            iwry = py + sin_theta*qx + cos_theta*qy + sin2_theta*0.31 + cos2_theta*0.09

                        #斜め45deg取付の場合　左
                        else:
                            # sin_theta = math.cos(theta - math.pi / 4)
                            # cos_theta = math.sin(theta - math.pi / 4)
                            # sin_theta = math.sin(theta - math.pi / 4) 
                            # cos_theta = math.cos(theta - math.pi / 4)
                            sin_theta = math.sin(-theta + math.pi / 4) 
                            cos_theta = math.cos(-theta + math.pi / 4)
                            sin2_theta = math.sin(-theta) 
                            cos2_theta = math.cos(-theta)
                            # iwrx = px + (qx+0.48) * cos_theta - (qy+0.05) * sin_theta #調整をかける, left
                            # iwry = py + (qx+0.48) * sin_theta + (qy+0.05) * cos_theta #調整をかける, left
                            iwrx = px + cos_theta*qx - sin_theta*qy + cos2_theta*0.48 - sin2_theta*0.05
                            iwry = py + sin_theta*qx + cos_theta*qy + sin2_theta*0.48 + cos2_theta*0.05
                            


                        #iwrx = px - qx * cos_theta + qy * sin_theta
                        #iwry = py 
                        # + qx * sin_theta + qy * cos_theta

                        amplitude = float(row[5])
                        timestanp = float(row[2])
                        v = float(row[6])

                        data_list.append([iwrx, iwry, amplitude, timestanp, qx, qy, v])

    # x, y データを取得
    x_list = [data[0] for data in data_list]
    y_list = [data[1] for data in data_list]
    amplitude_list = [data[2] for data in data_list]
    timestanp_list = [data[3] for data in data_list]
    q_x_list = [data[4] for data in data_list]
    q_y_list = [data[5] for data in data_list]
    v_list = [data[6] for data in data_list]

    # csv_file = 'C:/workspace/MELCO/data/melcomwr/230605_194737459_008/symbols.csv'
    #csv_file = 'C:\workspace\MELCO\data/20240123_VISON\MWR/240125_144504261_vison012503r/symbols80.csv'
    csv_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr+aqloc/"+finalfilename
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(x_list)):
            writer.writerow([x_list[i], y_list[i], amplitude_list[i], timestanp_list[i], q_x_list[i], q_y_list[i], v_list[i]])