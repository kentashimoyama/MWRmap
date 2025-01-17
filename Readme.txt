pythonファイル

EgoMotion.py　　　　　　　mwrデータに対し, amplitudeによる外れ値除去+azimuth角による外れ値除去+coscurvefitting
EgoMotion_mwr+aqloc.py　　mwr+aqlocデータに対し, amplitudeによる外れ値除去+azimuth角による外れ値除去+coscurvefitting(スイッチでon off変更可能)
gpggato19.py　　　　　　　aqlocデータを19系座標に変換
groundpoint_removal.py　　グラウンドポイントを除去。
mwrsplitcsv.py　　　　　　mwrデータをフレームごとに分割し別ファイルに保存
mwrsymbolcsv_finalver.py　mwrデータとaqlocデータを統合し, mwrを世界座標に変換.
mwrunixtoutc.py　　　　　 mwrデータの時刻をunixtimeからutctimeに変換
straightgeneration.py　　 各フレームについて直線近似
visualization.py　　　　　全フレームの直線を統合し, ビジュアライゼーション
separate_8sec.py          8秒ごとに点分割, 地面点群除去, amplitudeによる処理
execute_mvm.py            すべてをまとめて実行
config.py                 パラメータ、ディレクトリを設定

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
config.py →　gpggato19.py → mwrunixtoutc.py → mwrsplitcsv.py → mwrsymbolcsv_finalver.py → separate_8sec → groundpoint_removal.py → EgoMotion_mwr+aqloc.py → visualization.py
もしくは
config.py → gpggato19.py　→　execute_mvm.py									

LiDARによる喜久井町のリファレンス点群は以下に保存
https://waseda.box.com/s/lwp6gdfnd995rsbfmewvq9tace1quj37

フォルダ

aqloc                                    aqlocデータが入ってる
image0907　　　　　　　　　　　　　　　　mmsprobeでとった画像データが入ってる。モーションステレオで使おうとした
image0909　　　　　　　　　　　　　　　　同上
imagepoint　　　　　　　　　　　　　　　モーションステレオで構築した点群や画像間の対応点を投影したものや, 対応点の変化をベクトルで表現したものを投影した画像
MMSProbe　　　　　　　　　　　　　　　　モーションステレオで使用したMMSProbe.
motionstereo_result                     どうでもいい
mwr　　　　　　　　　　　　　　　　　　　mwrデータ
mwr+aqloc　　　　　　　　　　　　　　　　mwrとaqlocデータを複合したやつ。
other_data　　　　　　　　　　　　　　　たぶんmmsprobeでとった単独測位結果。結構どうでもいい
Structure-From=Motion-SFM--master　　　　一般人がつくったモーションステレオ用オープンソース
temp_using　　　　　　　　　　　　　　　自己位置推定時の8秒ごと地面点群除去処理の性能とミリ波ベクター作成性能を確認しようとしたときに使った
frame_mwr_aqloc                         mwr+aqloc点群を分割(cache)したデータ
groundpoint_filtered　　　　　　　　　　地面点群除去したデータ
amp_azimuth_filtered　　　　　　　　　　amp, azimuth処理したもの
wall　　　　　　　　　　　　　　　　　　壁を近似した直線

