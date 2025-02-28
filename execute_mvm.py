import pandas as pd
import numpy as np
import numpy as np
#import open3d as o3d
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Tkinterを使用する場合
import matplotlib.pyplot as plt
#from pclpy import pcl

from mwrunixtoutc import Mwrunix2utc
from mwrsplitcsv import MwrSplit
from mwrsymbolcsv_finalver import MwrSymbolCsv
from separate_8sec import SeparateMwr
from groundpoint_removal import GroundPointRemoval
from EgoMotion_mwr_aqloc import MwrProcessing
from visualization import Gen_map
from gpggato19 import Gpggato19

if __name__ == "__main__":        
    mwrunix2utc = Mwrunix2utc()
    ms = MwrSplit()
    msc = MwrSymbolCsv()
    sm = SeparateMwr()
    gp = GroundPointRemoval()
    mp = MwrProcessing()
    gm = Gen_map()
    gg = Gpggato19()


    ###### 入力箇所
    #mwr命名規則　最初は2024****で始まる、　最後に_${ミリ波データid}$rightwall${壁id}$をつける ex. 20240125_144508544_dat_3leftwall1
    #mwrローデータ
    # file_name_list = ["20250112_1736652806_adv_1rightwall0", "20250112_1736652808_adv_1leftwall0"]
    # file_name_list = ["20250112_1736656060_adv_2rightwall0", "20250112_1736656062_adv_2leftwall0"]
    # file_name_list = ["20241231_1735635244_adv_2rightwall0", "20241231_1735635247_adv_2leftwall0"]
    #file_name_list = ["20250120_1737336677_adv_1rightwall1", "20250120_1737336804_adv_1leftwall1"] 
    # file_name_list = ["20250120_1737337685_adv_2rightwall1", "20250120_1737337681_adv_2leftwall1"] 
    # file_name_list = ['20250120_1737352731_adv_3rightwall1', '20250120_1737352699_adv_3leftwall1', '20250120_1737352731_adv_3rightwall2']
    file_name_list = ['20250120_1737352699_adv_3leftwall2', '20250120_1737352731_adv_3rightwall3', '20250120_1737352699_adv_3leftwall3']
    #点群の積分を開始するフレームの時刻
    # whitelane_time_list = [1736652876.307, 1736652814.298]
    # whitelane_time_list = [1736656135.759, 1736656071.310]
    # whitelane_time_list = [1735635307.7, 1735635254.8]
    # whitelane_time_list = [1737337200.341, 1737337177.770]
    # whitelane_time_list = [1737337853.788, 1737337834.676]
    # whitelane_time_list = [1737353097.297,1737353075.128, 1737353148.327]
    whitelane_time_list = [1737353205.047,1737353239.001, 1737353239.001]
    #aqloc logファイルデータ
    # aqloclogfilename_list = ["20250112_1_map", "20250112_1_map"]
    # aqloclogfilename_list = ["20250112_2_map", "20250112_2_map"]
    #aql#oclogfilename_list = ["20250120_1_map", "20250120_1_map"]
    # aqloclogfilename_list =  ['20250120_2_map','20250120_2_map']
    aqloclogfilename_list =  ['20250120_3_map','20250120_3_map', '20250120_3_map']

    #左右mwr識別子. 例は, file_name_listの一番左のファイルが右mwrのデータ、真ん中のファイルが左Mwr, 一番右のファイルが右mwrデータであることを表す。
    #ex.) file_name_list = ["~_right", "~_left", "~_right"]
    #ex.) right_list = [True, False, True]
    right_list = [False, True, False]
    #####




    
   
    #logファイルを入力
    aqlocfilename_list = []
    for file_name in aqloclogfilename_list:
        gg.main(file_name)
        aqlocfilename_list.append(file_name+'_latlon19.csv')
    
    # mwr unixtimeからutctimeに変換
    for i, file_name in enumerate(file_name_list):
        print(f"processing mwr unix to utc ... input filename : {file_name}")

        mwrunix2utc.main(file_name, whitelane_time_list[i])


    
    # mwr点群フレームごとcsv分け
    input_filename_list = [a+"_utc.csv" for a in file_name_list]
    # input_filename_list = ["20240907_1725699643_adv_1right_utc.csv", "20240907_1725699657_adv_1left_utc.csv", "20240907_1725700271_adv_2right_utc.csv", "20240907_1725700273_adv_2left_utc.csv", "20240907_1725700578_adv_3right_utc.csv", "20240907_1725700579_adv_3left_utc.csv"]
    # outputdirname_list = ["csv_0907_1right", "csv_0907_1left", "csv_0907_2right", "csv_0907_2left", "csv_0907_3right", "csv_0907_3left"]
    outputdirname_list = ["csv_"+a[4:8]+"_"+ a.split("_")[-1] for a in file_name_list]

    for idx, input_filename in enumerate(input_filename_list):
        print(f"processing mwr split ... input filename : {input_filename}")
        output_dir =  outputdirname_list[idx]

        ms.main(input_filename, output_dir)


    
    # mwr+aqloc
    # iwrdirname_list = ["csv_0907_1left", "csv_0907_2left", "csv_0907_3left", "csv_0907_1right", "csv_0907_2right", "csv_0907_3right"]
    iwrdirname_list = outputdirname_list
    # finalfilename_list = ["symbols_0907_1left.csv", "symbols_0907_2left.csv", "symbols_0907_3left.csv", "symbols_0907_1right.csv", "symbols_0907_2right.csv", "symbols_0907_3right.csv"]
    finalfilename_list = ["symbols_" + b[4:] + ".csv" for b in outputdirname_list]
    
    for iwrdirname, aqlocfilename, finalfilename, right in zip(iwrdirname_list, aqlocfilename_list, finalfilename_list, right_list):
        msc.main(iwrdirname, aqlocfilename, finalfilename, right)


    
    # mwrcache
    # input_filename_list = ["symbols_0907_1right.csv", "symbols_0907_1left.csv"]
    input_filename_list = finalfilename_list
    for input_filename in input_filename_list:
        sm.sep_8sec(input_filename)


    
    # 地面点群除去
    # csv_dirname_list = ["symbols_0907_1left", "symbols_0907_2left", "symbols_0907_3left", "symbols_0907_1right", "symbols_0907_2right", "symbols_0907_3right"]
    csv_dirname_list = [c.split(".")[0] for c in finalfilename_list]
    visualize = False

    for csv_dirname in csv_dirname_list:
        # CSVファイルを読み込む
        print(f"process {csv_dirname} ...")

        gp.main(csv_dirname, visualize)

    # amp, azimuth閾値
    cosfitting = False
    is_MELCO = 1 #MELCO製:1 not MELCO製(TI製?):0
    # dir_list = ["symbols_0907_1left", "symbols_0907_2left", "symbols_0907_3left", "symbols_0907_1right", "symbols_0907_2right", "symbols_0907_3right"]
    dir_list = csv_dirname_list

    for dir in dir_list:
        mp.main(dir, is_MELCO, cosfitting)


    #ransac直線近似
    mwr_all_filename_list = [d+".csv" for d in csv_dirname_list]
    mwr_filename_list = csv_dirname_list
    aqloc_filename_list = aqlocfilename_list
    for idx, mwr_filename in enumerate(mwr_filename_list):
        gm.main(mwr_all_filename_list[idx], mwr_filename, aqloc_filename_list[idx])
