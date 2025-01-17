import csv
import codecs
from pyproj import Transformer
import math
import numpy as np
from config import Set_config

class Gpggato19():
    def __init__(self):
        sc = Set_config()
        self.home_dir = sc.home_dir
        self.epsg = sc.epsg

        # WGS84 to JGD2011
        wgs84_to_jgd2011 = Transformer.from_crs('EPSG:4326', 'EPSG:6668')

        # EPSG:4326 to EPSG:6677
        # epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6674")
        self.epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", f"epsg:{self.epsg}")
        # 6系三重県vison:epsg:6674, 9系東京都早稲田大学大学喜久井町キャンパス, 神奈川県三菱電機鎌倉製作所KTF:epsg:6677

    # GPGGA latitude, longitude data --> 19 latitude, longitude
    def convert_gpgga_to_jgd2011(self, latitude, longitude):
        # concert GPGGA dddmm.mmmm to ddd.dddd
        wgs84_latitude = float(latitude[:2]) + float(latitude[2:]) / 60
        wgs84_longitude = float(longitude[:3]) + float(longitude[3:]) / 60

        # jgd2011_longitude, jgd2011_latitude = wgs84_to_jgd2011.transform(wgs84_longitude, wgs84_latitude)
        # latitude19, longitude19 = epsg4326_to_epsg6677.transform(jgd2011_latitude, jgd2011_longitude)
        latitude19, longitude19 = self.epsg4326_to_epsg6677.transform(wgs84_latitude, wgs84_longitude)

        return latitude19, longitude19

    def cal_deg(self, lat, lon, pre_lat, pre_lon, delta_t):
            #if t_pre == 0:
                #return
            
            vx = (float(lat) - float(pre_lat))/delta_t
            vy = (float(lon) - float(pre_lon))/delta_t
            
            yaw=math.atan2(vy,vx)
            
            return yaw


    def compute_direction_vector(self, point1, point2):
        """
        2つの点の間の方向ベクトルを計算する
        """
        return np.array(point2) - np.array(point1)

    def compute_yaw_angle(self, vector1, vector2):
        """
        2つのベクトル間のヨー角を計算する
        """
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos_theta = dot_product / norm_product
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 角度をラジアンで計算
        return angle #np.degrees(angle)  # ラジアンを度に変換

    def cal_deg2(self, pro_lat, pro_lon, lat, lon, pre_lat, pre_lon, pro_delta_t, delta_t):
        """
        3点の位置データを用いて進行方向（ヨー角）を推定する
        """
        # point_prev = [pre_lat, pre_lon]
        # point_curr = [lat, lon]
        # point_next = [pro_lat, pro_lon]

        # vector1 = compute_direction_vector(point_prev, point_curr)
        # vector2 = compute_direction_vector(point_curr, point_next)
        
        # # 進行方向ベクトルの平均を計算
        # direction_vector = (vector1 + vector2) / 2.0
        
        # # ヨー角を計算
        # yaw = compute_yaw_angle(vector1, vector2)

        dt_minus1 = delta_t
        F_t1_x = pro_lat
        F_t0_x = lat
        dt1 = pro_delta_t
        F_t_minus1_x = pre_lat
        F_t1_y = pro_lon
        F_t0_y = lon
        F_t_minus1_y = pre_lon


        numerator_x = (dt_minus1**2 * (F_t1_x - F_t0_x)) + (dt1**2 * (F_t0_x - F_t_minus1_x))
        numerator_y = (dt_minus1**2 * (F_t1_y - F_t0_y)) + (dt1**2 * (F_t0_y - F_t_minus1_y))
        denominator = (dt_minus1**2 * dt1) + (dt1**2 * dt_minus1)
        vx = numerator_x / denominator
        vy = numerator_y / denominator
        yaw=math.atan2(vy,vx)

        return yaw
        
        # 結果を返す
        return yaw, direction_vector

    def main(self, file_name):
        # file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc/"+file_name
        file = self.home_dir + "\\aqloc\\"+file_name

        # file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\AQLOC/012503"


        with open(file+".log", "r") as log_file, open(file+".csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            for line in log_file:
                parts = line.strip().split(",")
                writer.writerow(parts)

        # open file
        #with codecs.open('C:/workspace/MELCO/data/20240123_VISON/AQLOC/012504.csv', 'r', encoding='utf-8-sig') as csvfile:
        with codecs.open(file+'.csv', 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            # rows = [row for row in reader]
            rows_o = [row for row in reader]
            rows = []
            for i, row in enumerate(rows_o):
                if "GPGGA" in rows_o[i][0]:
                    rows.append(row)

        # save file
        #with open('C:/workspace/MELCO/data/20240123_VISON/AQLOC/012504_19.csv', 'w', encoding='utf-8-sig') as f:
        with open(file+'_latlon19.csv', 'w', encoding='utf-8-sig') as f:
            writer = csv.writer(f, lineterminator='\n')
            for i, row in enumerate(rows):
                if "GPGGA" in rows[i][0]:
                    if i > 0:
                        print(i)
                        time = rows[i][1] # time
                        latitude = rows[i][2] # GPGGAlatitude
                        longitude = rows[i][4] # GPGGAlongitude
                    

                        # convert GPGGA data --> 19
                        lat, lon = self.convert_gpgga_to_jgd2011(latitude, longitude)

                        if i == 1:
                            row = [time, lat, lon]
                            writer.writerow(row)
                            continue

                        if i == len(list(rows))-1:
                            continue
                        
                        pro_time = rows[i+1][1]
                        pro_latitude = rows[i+1][2]
                        pro_longitude = rows[i+1][4]

                        pre_time = rows[i-1][1] # time
                        pre_latitude = rows[i-1][2] # GPGGAlatitude
                        print(f"pre:{pre_latitude}")
                        pre_longitude = rows[i-1][4] # GPGGAlongitude
                            
                        pre_lat, pre_lon = self.convert_gpgga_to_jgd2011(pre_latitude, pre_longitude)
                        pro_lat, pro_lon = self.convert_gpgga_to_jgd2011(pro_latitude, pro_longitude)

                        delta_t = float(time)-float(pre_time)
                        pro_delta_t = float(pro_time) - float(time)
                        #yaw = self.cal_deg(lat, lon, pre_lat, pre_lon, delta_t)
                        yaw = self.cal_deg2(pro_lat, pro_lon, lat, lon, pre_lat, pre_lon, pro_delta_t, delta_t)
                        
                        row = [time, lat, lon, yaw]
                        writer.writerow(row)

                # #yaw caliculated by aqloc
                # if "PMSBATV" in rows[i][0]:
                #     if i > 1:
                #         print(i)
                #         roll = float(rows[i][4])
                #         roll = (roll*math.pi)/180
                #         pitch = float(rows[i][5])
                #         pitch = (pitch*math.pi)/180
                #         yaw = float(rows[i][6])
                #         yaw = (yaw*math.pi)/180
                #         row = [time, lat, lon, yaw, roll, pitch]
                #         writer.writerow(row)



if __name__ == "__main__":
    gg = Gpggato19()
    
    #logファイルを入力
    file_name = "25016_12"
    
    gg.main(file_name)










