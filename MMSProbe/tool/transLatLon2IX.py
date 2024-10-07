from pyproj import Transformer
import csv

# epsg6677_to_epsg4326 = Transformer.from_crs("epsg:6677", "epsg:4326")
epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6677")
latlon=[]
# with open(r'E:\workspace\MMSProbe_2021-20211109\data\results\debug_202202220731\Maingps_raw.csv') as f:
with open(r'E:\MELCO\data\vison\MMSProbe\221130040016/GPSData.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        latlon.append([row[1], row[2]])

with open(r'E:\MELCO\data\vison\MMSProbe\221130040016/GPSData_IX.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i, row in enumerate(latlon):
        if i>0:
            lat, lon = epsg4326_to_epsg6677.transform(float(latlon[i][0]), float(latlon[i][1]))
            row = [lat, lon]
            writer.writerow(row)
            print(lat, lon)
