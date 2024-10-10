EgoMotion.py　　　　　　　mwrデータに対し, amplitudeによる外れ値除去+azimuth角による外れ値除去+coscurvefitting
EgoMotion_mwr+aqloc.py　　mwr+aqlocデータに対し, amplitudeによる外れ値除去+azimuth角による外れ値除去+coscurvefitting
gpggato19.py　　　　　　　aqlocデータを19系座標に変換
groundpoint_removal.py　　グラウンドポイントを除去。
mwrsplitcsv.py　　　　　　mwrデータをフレームごとに分割し別ファイルに保存
mwrsymbolcsv.py　　　　　 mwrデータとaqlocデータを統合し, mwrを世界座標に変換.
mwrunixtoutc.py　　　　　 mwrデータの時刻をunixtimeからutctimeに変換
straightgeneration.py　　 各フレームについて直線近似
visualization.py　　　　　全フレームの直線を統合し, ビジュアライゼーション

以下モーションステレオ関係。
calc_wc_gen_ortho.py      未完成ファイル。
generate_gif.py           mmsprobeカメラ画像からgifを作成する。
motionstereo.py　　　　　 openMVGを使ってsfmを行うようのファイル。デバグが済んでおらず動かないが, 次モーションステレオをやるならこれを動かしてみるのが早いと思われる。
orthoimage_mwrpoints.py   自分でモーションステレオをやってみようとしたやつ。精度が悪すぎてできなかったので使わんほうがいい。
plt_timeorder.py          mwr点群を時系列順にプロットする動画を作成する。
plt_yaw.py                aqloc位置情報から求めたyaw角と車両走行軌跡を描画
write_camerapose.py       カメラパラメータからカメラのロールピッチよーを時系列的に可視化する。
write_pose.py             aqlocの時系列位置情報から、yaw角を推定し時系列でグラフに可視化



MWRマップ作製時のファイル実行順
gpggato19.py → mwrunixtoutc.py → mwrsplitcsv.py → mwrsymbolcsv_finalver.py → groundpoint_removal.py → EgoMotion_mwr+aqloc.py → straightgeneration.py

LiDARによる喜久井町のリファレンス点群は以下に保存
https://waseda.box.com/s/lwp6gdfnd995rsbfmewvq9tace1quj37
