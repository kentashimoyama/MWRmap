##ミリ波レーダと画像の座標系はどのように合わせるか．
#1. カメラ姿勢の推定→姿勢を加味した補正用の画像変換行列の算出→座標系変換行列の算出→変換行列によるピクセルごとの世界座標(原点カメラの系)の推定
#2. 逆変換行列による指定した4点の世界座標から画像座標への変換→オルソ化

##カメラ姿勢の推定byカメラキャリブレーション, 変換行列の算出，変換行列によるピクセルごとの世界座標推定

import cv2
import numpy as np

# キャリブレーションパターンのサイズ（チェッカーボードの内側のコーナー数）
pattern_size = (9, 6)

# チェッカーボードの1つの正方形のサイズ（単位はメートル）
square_size = 0.025

# キャリブレーション画像のパス
calibration_images = ['calib1.jpg', 'calib2.jpg', 'calib3.jpg', 'calib4.jpg', 'calib5.jpg']

# ワールド座標系のチェッカーボードのコーナー点を準備
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 画像座標系とワールド座標系でのチェッカーボードのコーナー点
objpoints = []
imgpoints = []

# キャリブレーション画像からコーナー点を検出
for fname in calibration_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# カメラキャリブレーション
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 回転ベクトルを回転行列に変換
R, _ = cv2.Rodrigues(rvecs[0])





# 回転行列Rをオイラー角に変換する関数
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# オイラー角を回転行列に変換する関数
def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# # 例として、回転行列Rを設定（実際にはキャリブレーションなどで取得する）
# R = np.array([[0.9998477, 0.0174524, 0.0000000],
#               [-0.0174524, 0.9998477, 0.0000000],
#               [0.0000000, 0.0000000, 1.0000000]])

# 回転行列からオイラー角を取得（ラジアン）
euler_angles = rotation_matrix_to_euler_angles(R)
print("Euler angles (radians):", euler_angles)
print("Euler angles (degrees):", np.degrees(euler_angles))

# 画像の読み込み
input_image = cv2.imread('drive_recorder_image.jpg')

# 回転行列を使って画像を補正
def correct_image(image, euler_angles):
    h, w = image.shape[:2]
    f = 500  # 仮の焦点距離
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]])  # カメラ行列

    # オイラー角から回転行列を計算
    R = euler_angles_to_rotation_matrix(euler_angles)

    # 逆回転行列を計算
    R_inv = np.linalg.inv(R)

    # ホモグラフィ行列を計算
    H = K @ R_inv @ np.linalg.inv(K)

    # 画像をアフィン変換
    corrected_image = cv2.warpPerspective(image, H, (w, h))
    return corrected_image

corrected_image = correct_image(input_image, euler_angles)

# 画像の表示
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# ピクセル座標をカメラ座標系に変換する関数
def pixel_to_camera(pixel_coords, mtx, dist):
    # 逆投影のために歪み補正
    pixel_coords_undist = cv2.undistortPoints(np.array([pixel_coords], dtype=np.float32), mtx, dist, P=mtx)
    # カメラ座標系に変換
    camera_coords = np.array([pixel_coords_undist[0][0][0], pixel_coords_undist[0][0][1], 1.0])
    return camera_coords

# カメラ座標系を世界座標系に変換する関数
def camera_to_world(camera_coords, R, tvec):
    # カメラ座標系を世界座標系に変換
    world_coords = np.dot(np.linalg.inv(R), (camera_coords - tvec).reshape(-1, 1))
    return world_coords

# # カメラの姿勢角を取得するための関数
# def rotation_matrix_to_euler_angles(R):
#     sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6
#     if not singular:
#         x = np.arctan2(R[2, 1], R[2, 2])
#         y = np.arctan2(-R[2, 0], sy)
#         z = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         x = np.arctan2(-R[1, 2], R[1, 1])
#         y = np.arctan2(-R[2, 0], sy)
#         z = 0
#     return np.array([x, y, z])

# 例: 画像上のピクセル座標 (u, v) を世界座標に変換
u, v = 500, 300
camera_coords = pixel_to_camera((u, v), mtx, dist)
world_coords = camera_to_world(camera_coords, R, tvecs[0])

print("Camera coordinates:", camera_coords)
print("World coordinates:", world_coords)

# # 姿勢角（オイラー角）を取得
# euler_angles = rotation_matrix_to_euler_angles(R)
# print("camera Euler angles (radians):", euler_angles)
# print("camera Euler angles (degrees):", np.degrees(euler_angles))



##オルソ画像の作成
# # カメラキャリブレーションパラメータ
# mtx = np.array([[fx, 0, cx],
#                 [0, fy, cy],
#                 [0, 0, 1]])
# dist = np.array([k1, k2, p1, p2, k3])
# R = np.array([[r11, r12, r13],
#               [r21, r22, r23],
#               [r31, r32, r33]])
# tvec = np.array([tx, ty, tz])




## 入力
# 世界座標系の4点（例）
world_points = np.array([
    [X1, Y1, Z1],
    [X2, Y2, Z2],
    [X3, Y3, Z3],
    [X4, Y4, Z4]
], dtype=np.float32)

# 世界座標から画像座標への逆変換　変換行列を見直しておく
def world_to_pixel(world_points, mtx, dist, R, tvec):
    # 回転ベクトルに変換
    rvec, _ = cv2.Rodrigues(R)
    # 世界座標を画像座標に変換
    img_points, _ = cv2.projectPoints(world_points, rvec, tvec, mtx, dist)
    return img_points.reshape(-1, 2)

# 画像座標を取得
image_points = world_to_pixel(world_points, mtx, dist, R, tvecs)

# オルソ画像のサイズ　決め方検討．ピクセルと実寸の対応によって決めるのが良いか．
width, height = 500, 500

# オルソ画像の4点の座標（例: 正方形）
ortho_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# ホモグラフィ行列を計算
H, _ = cv2.findHomography(image_points, ortho_points)

# 入力画像の読み込み
input_image = cv2.imread('input.jpg')

# オルソ画像の作成
ortho_image = cv2.warpPerspective(input_image, H, (width, height))

# オルソ画像の表示
cv2.imshow('Ortho Image', ortho_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
