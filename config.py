class Set_config():
    def __init__(self):
        self.mwr_gnss_offset_x_right = 0.265 #mwrとgnss受信機のx方向(横方向)距離
        self.mwr_gnss_offset_y_right = 0.29 #mwrとgnss受信機のy方向(奥行方向)距離
        self.mwr_gnss_offset_x_left = 0.515
        self.mwr_gnss_offset_y_left = 0.25
        self.home_dir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp" #このファイルが入っているディレクトリ
        self.amp_th = 65 #amplitude閾値
        self.gpth = 15 #投票処理閾値 vison15, 喜久井町10
        self.voxel_size = 0.36 #投票処理grid大きさ
        self.timeseg = 8 #点群cache時間
        self.epsg = 6677 #vison6674, 喜久井町6677