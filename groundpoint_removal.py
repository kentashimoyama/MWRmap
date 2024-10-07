import numpy as np
#import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
#from pclpy import pcl

def filter_ground_and_vegetation(points, voxel_size=0.05, threshold=5):
    # グリッドのサイズを設定
    grid_size = voxel_size

    # 点群の最小値と最大値を求める
    min_x, min_y, _, _, _, _, _ = np.min(points, axis=0)
    max_x, max_y, _, _, _, _, _ = np.max(points, axis=0)

    # グリッドの範囲を計算
    grid_min_x = min_x
    grid_min_y = min_y
    grid_max_x = max_x
    grid_max_y = max_y

    # グリッドの数を計算
    grid_num_x = int(np.ceil((grid_max_x - grid_min_x) / grid_size)) #少数第二位を切り上げ
    grid_num_y = int(np.ceil((grid_max_y - grid_min_y) / grid_size))

    # グリッド内の点の数を計算
    grid_count = np.zeros((grid_num_x, grid_num_y)) #各gridに含まれる点の数を保存する配列
    points_ingrid = np.empty((grid_num_x, grid_num_y), dtype=object) #各gridの点の座標を格納する配列
    for i in range(grid_num_x):
        for j in range(grid_num_y):
            points_ingrid[i, j] = [] #配列をlistで初期化

    for num, point in enumerate(points):
        print(f"generating grid ...  {num+1}/{points.shape[0]}")
        x, y, amp, t, xrange, yrange, velocity = point
        grid_x = int((x - grid_min_x) // grid_size) #x方向grid index. 1つめのgridのインデックスは0．
        grid_y = int((y - grid_min_y) // grid_size) #y方向grid index. 1つめのgridのインデックスは0．
        if 0 <= grid_x < grid_num_x and 0 <= grid_y < grid_num_y:
            grid_count[grid_x, grid_y] += 1 #gridを表す配列に各gridの点の個数を保存．
            # points_ingrid[grid_x, grid_y].append([x, y]) #x, y座標を保存. 2次元配列の1要素が1グリッドにあたる． [[ [[x,y], [x,y]] ], [ [] ], [ [[x,y]] ]]
            points_ingrid[grid_x, grid_y].append([x, y, amp, t, xrange, yrange, velocity]) #x, y座標を保存. 2次元配列の1要素が1グリッドにあたる． [[ [[x,y,amp,t], [x,y,amp,t]] ], [ [] ], [ [[x,y,amp,t]] ]]


    indices = np.where(grid_count >= threshold) #閾値より大きい??　小さい??  gridのindexのみ抽出．出力：[[行index1, 行index2, ...], [列index1, 列index2, ...]]
    #print(points_ingrid.shape)

    filtered_points_list = points_ingrid[list(indices[0]), list(indices[1])] #1次元numpy配列で出力される. 抽出されたgridに入っている点の座標 [　[[x, y, amp, t], [x,y,amp,t]], [x,y,amp,t], [[x, y,amp,t], [x,y,amp,t],　...　] 
    print(f"extracted grids: {filtered_points_list.shape[0]}")

    filtered_ps_list = []
    for i in range(filtered_points_list.shape[0]):
        for filtered_p_list in filtered_points_list[i]:
            filtered_ps_list.append(filtered_p_list)
    filtered_points = np.array(filtered_ps_list)
    filtered_points = filtered_points[filtered_points[:, 3].argsort()] #unixtimeの列を基準にソート

    # 直線を描画 open3dを使用。
    #line_set = grid_line(grid_min_x, grid_min_y ,grid_max_x, grid_max_y, grid_num_x, grid_num_y, grid_size)

    return filtered_points



def grid_line(grid_min_x, grid_min_y ,grid_max_x, grid_max_y, grid_num_x, grid_num_y, grid_size):
    # 直線の端点
    line_points_list = [[grid_min_x, grid_min_y, 0], [grid_min_x, grid_max_y, 0]] #[始点1, 終点1, 始点2, 終点2, ...]
    for i in range(grid_num_x):
        line_points_list.append([grid_min_x + grid_size*(i+1), grid_min_y, 0]) #始点
        line_points_list.append([grid_min_x + grid_size*(i+1), grid_max_y, 0]) #終点
    line_points_list.append([grid_min_x, grid_min_y, 0])
    line_points_list.append([grid_max_x, grid_min_y, 0])
    for j in range(grid_num_y):
        line_points_list.append([grid_min_x, grid_min_y + grid_size*(i+1), 0]) #始点
        line_points_list.append([grid_max_x, grid_min_y + grid_size*(i+1), 0]) #終点
    line_points = np.array(line_points_list, dtype=np.float64)
    print(line_points)

    # 直線の描画
    line_set = o3d.geometry.LineSet()
    #line_set.points = o3d.utility.Vector3dVector(line_points)
    #line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector([
    [i*2, i*2+1] for i in range(line_points.shape[0])   
    ])
    line_set.paint_uniform_color([0, 0, 1])  # 青色で描画

    return line_set

# # Open3Dを用いて点群をグリッドに分割し、閾値に基づいてフィルタリングする関数
# def filter_ground_and_vegetation_o3d(points, voxel_size=0.05, threshold=10):
#     # Open3DのPointCloudオブジェクトに変換
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     # ボクセルグリッドフィルターを適用
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
#     print(voxel_grid.get_voxels())
#     # print(111111111111111)
#     # 各ボクセル内の点の数をカウント
#     voxel_density = {}
#     for voxel in voxel_grid.get_voxels():
#         print(voxel.grid_index)
#         voxel_density[voxel.grid_index] = voxel_density.get(voxel.grid_index, 0) + 1
    
#     # フィルタリング
#     filtered_points = []
#     for point in np.asarray(pcd.points):
#         grid_index = tuple((point / voxel_size).astype(int))
#         if voxel_density.get(grid_index, 0) < threshold:
#             filtered_points.append(point)
    
#     # フィルタリング結果をDataFrameに変換
#     filtered_points_df = pd.DataFrame(filtered_points, columns=['x', 'y'])
    
#     return filtered_points_df


# # PCLを用いて点群をグリッドに分割し、閾値に基づいてフィルタリングする関数
# def filter_ground_and_vegetation_pcl(points, grid_size=0.05, threshold=10):
#     # PCLのPointCloudオブジェクトに変換
#     pc = pcl.PointCloud.PointXYZ()
#     pc.from_array(points[['x', 'y']].values.astype(np.float32))
    
#     # ボクセルグリッドフィルターを設定
#     voxel_grid = pcl.filters.VoxelGrid.PointXYZ()
#     voxel_grid.setInputCloud(pc)
#     voxel_grid.setLeafSize(grid_size, grid_size, 1.0)  # z方向のサイズは無視

    
    
#     # フィルタリングを実行
#     filtered_pc = pcl.PointCloud.PointXYZ()
#     voxel_grid.filter(filtered_pc)
    
#     # フィルタリングした点群をNumPy配列に変換
#     filtered_points = np.asarray(filtered_pc.xyz)
    
#     # 各グリッドの点数をカウント
#     grid_index = ((filtered_points[:, :2] / grid_size).astype(int))
#     unique, counts = np.unique(grid_index, axis=0, return_counts=True)
    
#     # 閾値に基づいてフィルタリング
#     mask = counts < threshold
#     valid_grids = unique[mask]
    
#     # 有効なグリッドに属する点を抽出
#     valid_mask = np.isin(grid_index, valid_grids).all(axis=1)
#     filtered_points = points[valid_mask]
    
#     return filtered_points






# CSVファイルのパス
#csv_file_path = r'C:\Users\shimo\PycharmProjects\symbols.csv'
#csv_file_path = r"C:\Users\shimo\PycharmProjects\MELCO引継ぎ用\簡易地図生成\mwr+aqloc\symbols.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_0717_tennis_1_left_80azi_cosinlier.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_0717_tennis_1_left.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_tennis.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_0627_6_mwr1_80azi_cosinlier.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_0717_agawall_1_left_80azi_cosinlier.csv"
# csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\symbols_0717_tennis_1_withoutlier_right.csv"
# csv_file_path_list = [r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_1left.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_1right.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_2left.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_2right.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_3left.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_3right.csv"]
csv_file_path_list = [r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_2left.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\symbols_0907_2right.csv"]






for csv_file_path in csv_file_path_list:
    # CSVファイルを読み込む
    print(f"process {csv_file_path} ...")
    df = pd.read_csv(csv_file_path, header=None)  # header=None はヘッダーがない場合に使用します

    # 0列目と1列目のデータを抽出する
    extracted_columns = df.iloc[:, [0, 1]] #dataframe型 x, y

    # 抽出したデータを表示
    #print(extracted_columns)
    # world_pos = np.array(extracted_columns) #[[x, y], [x, y], [x, y]...]
    world_pos = np.array(df)  #[[x, y, amp, t, Xrange, Yrange, Velocity[m/s]], [x, y, amp, t, , Xrange, Yrange, Velocity[m/s]], [x, y, amp, t, , Xrange, Yrange, Velocity[m/s]]...]

    #print(world_pos)

    #open3dによる元点群の可視化
    # pcd_origin = o3d.geometry.PointCloud()
    # pcd_origin.points = o3d.utility.Vector3dVector(world_pos)
    # pcd_origin.paint_uniform_color([1, 0, 0])
    #o3d.visualization.draw_geometries([pcd])

    # 点群のフィルタリング
    th = 2
    filtered_points = filter_ground_and_vegetation(world_pos, voxel_size=0.05, threshold=th) #gridあたり何点以上 大体閾値は2

    # フィルタリング結果を表示
    print(f"extracted points: {filtered_points.shape[0]}")

    # filtered_points = np.column_stack((filtered_points, np.zeros(filtered_points.shape[0], dtype=int)))

    # Open3Dを使ってフィルタリング結果を可視化
    # pcd_filterd = o3d.geometry.PointCloud()
    # pcd_filterd.points = o3d.utility.Vector3dVector(filtered_points)
    # pcd_filterd.paint_uniform_color([0, 1, 0])
    #o3d.visualization.draw_geometries([pcd_filterd])

    # pltによる元の点群の可視化
    plt.subplot(1, 2, 1)  # 1行2列のレイアウトで1番目のプロット
    plt.scatter(world_pos[:, 0], world_pos[:, 1], s=1, c='blue', alpha=0.5, label='mwr_points')
    plt.title('azimuth_amp_coscurve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    grid_size_x = 0.05
    grid_size_y = 0.05
    # plt.xticks(np.arange(-32720, -32680, grid_size_x))
    # plt.yticks(np.arange(-10155, -10105, grid_size_y))
    plt.grid(True)  # グリッドの表示


    # pltを使ってフィルタリング結果を可視化
    plt.subplot(1, 2, 2)  # 1行2列のレイアウトで2番目のプロット
    plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1, c='blue', alpha=0.5, label='mwr_points')
    plt.title('azimuth_amp_coscurve_vote')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    grid_size_x = 0.05
    grid_size_y = 0.05
    # plt.xticks(np.arange(-32720, -32680, grid_size_x))
    # plt.yticks(np.arange(-10155, -10105, grid_size_y))
    plt.grid(True)  # グリッドの表示
    #plt.tight_layout()  # レイアウトを自動調整
    plt.show()

    # 新しいCSVファイルを作成、保存
    extracted_mwr_df = pd.DataFrame(filtered_points)
    extracted_mwr_df.to_csv(csv_file_path.split(".")[0]+"_extracted_"+str(th)+".csv"  , index=False, header=False)



    # #lidar点群を可視化
    # # PCDファイルの読み込み

    # # 点群データの取得
    # points = np.asarray()

    # # 0, 1, 2列目(X, Y, Z)のみ使用する
    # new_points = points[:, 0:3]

    # print(pcd)
    # # オフセットの量
    # offset_x = -32728.7501
    # offset_y = -10103.2483
    # offset_z = 14

    # # 点群のオフセット
    # new_points[:, 0] += offset_x
    # new_points[:, 1] += offset_y
    # new_points[:, 2] += offset_z

    # # 可視化
    # pcd_lidar = o3d.geometry.PointCloud()
    # pcd_lidar.points = o3d.utility.Vector3dVector(new_points)
    # pcd_lidar.paint_uniform_color([0, 1, 0])



    # まとめて描画
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # #vis.add_geometry(pcd_origin)
    # #vis.add_geometry(pcd_filterd)
    # # vis.add_geometry(line_set)
    # #vis.add_geometry(pcd_lidar)
    # vis.run()
    # vis.destroy_window()