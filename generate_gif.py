import os
from PIL import Image

# 画像を保存しているディレクトリのパス
input_directory = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\image0907\image0907/"
output_gif_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\document/0907_gif.gif"

# JPEG画像ファイルのリストを取得
images = []
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_directory, filename)
        images.append(Image.open(image_path))

# GIFを保存
if images:
    images[0].save(output_gif_path,                                #save_all:複数フレームを持つgifとして保存, append_images[1:]:gifに画像を追加, optimize:False:サイズの最適化, ループ回数：0は無限
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=1000,  # フレームの表示時間（ミリ秒）
                   loop=0)        # 無限ループ
    print(f"GIFが作成されました: {output_gif_path}")
else:
    print("指定されたディレクトリにJPEG画像が見つかりませんでした。")