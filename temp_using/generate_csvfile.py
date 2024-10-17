import pandas as pd

# CSVファイルを読み込む
input_file = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144504261_vison012503r/240125_144504261_dat_utc.csv'  # 読み込むCSVファイルのパス
# input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\MMSProbe_2024\data\20240123_VISON\MWR\240125_144508544_vison012503l/240125_144508544_dat_utc.csv"
output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr/"+input_file.split("/")[1].split(".")[0]+"_2.csv"  # 保存するCSVファイルのパス

# DataFrameを作成
df = pd.read_csv(input_file)

# 指定された列の順番で並べ替え
columns_to_select = [0, 10, 14, 15, 8, 4, 13]
df_reordered = df.iloc[:, columns_to_select]
new_header = ["FrameID", "UnixTime", "Xrange", "Yrange", "Amplitude", "Velocity[m/s]", "UTC"]
df_reordered.columns = new_header
df_reordered.insert(1, 'SymbolID', 0)

# 新しいCSVファイルに保存
df_reordered.to_csv(output_file, index=False)
