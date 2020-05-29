import math

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import randint
from PIL import Image, ImageDraw


def discriminant_analysis(input_file_path):
    img = cv2.imread(input_file_path)
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ヒストグラムを作成
    histgram = [0] * 256
    for i in range(0, len(gray)):
        for j in range(0, len(gray[0])):
            histgram[gray[i][j]] += 1

    max_t = max_val = 0

    # 判別分析法を使って2値化
    for t in range(0, 256):
        # 画素数
        w1 = w2 = 0
        # クラス別合計値
        sum1 = sum2 = 0
        # クラス別平均
        m1 = m2 = 0.0

        for i in range(0, t):
            w1 += histgram[i]
            sum1 += i * histgram[i]

        for j in range(t, 256):
            w2 += histgram[j]
            sum2 += j * histgram[j]

        # 0除算を防ぐ
        if w1 == 0 or w2 == 0:
            continue
        
        # クラス別平均の算出
        m1 = sum1 / w1
        m2 = sum2 / w2

        # 結果を算出
        result = w1 * w2 * (m1 - m2) * (m1 - m2)

        if max_val < result:
            max_val = result
            max_t = t

    for i in range(0, len(gray)):
        for j in range(0, len(gray[0])):
            gray[i][j] = 0 if gray[i][j] < max_t else 255

    return gray


def k_mean(original_img, k):
    #STEP1
    w, h = original_img.size

    # 画像を扱いやすい２次配列に変換
    img_pixels = np.array([[img.getpixel((x,y)) for x in range(w)] for y in range(h)])
    # 減色画像用の２次配列も用意しておく
    reduce_img_pixels = np.array([[(0, 0, 0) for x in range(w)] for y in range(h)])

    # 代表色の初期値をランダムに設定
    class_values = []
    for i in range(k):
        class_values.append(np.array([randint(256), randint(256), randint(256)]))

    #STEP2
    # 20回繰り返す
    for i in range(5):
        #STEP2-1
        print("ranning at iteration No." + str(i))
        sums = []
        for i in range(k):
            sums.append(np.array([0, 0, 0])) 
            sums_count = [0] * k

        #STEP2-2
        # 各画素のクラスを計算
        for x in range(w):
            for y in range(h):
                min_d = (256 ** 2) * 3
                class_index = 0
                # 一番近い色（クラス）を探索
                for j in range(k):
                    d = sum([x*x for x in img_pixels[y][x] - class_values[j]])
                    if min_d > d:
                        min_d = d
                        class_index = j
                sums[class_index] += img_pixels[y][x]
                sums_count[class_index] += 1
                reduce_img_pixels[y][x] = tuple(list(map(int, class_values[class_index])))

        #STEP2-3
        # 代表色を更新
        for m in range(k):
            class_values[m] = sums[m] / sums_count[m]

    # STEP3
    # ２次元配列から加工後の画像へ変換
    reduce_img = Image.new('RGB', (w, h))
    for x in range(w):
        for y in range(h):
            reduce_img.putpixel((x, y), tuple(reduce_img_pixels[y][x]))

    return reduce_img


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        print(i)
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator,
        cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]]
    )
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # file名の設定
    file_name = 'test.jpg'
    type = ''
    input_file_path = lambda file_name: './input/{}'.format(file_name)
    output_file_path = lambda type, file_name: './output/{}_{}'.format(type, file_name)

    # 判別分析法
    type = 'judge'
    gray = discriminant_analysis(input_file_path(file_name))
    cv2.imwrite(output_file_path(type, file_name), gray)

    # kmeans法
    ## 画像ファイルの読み込み
    type = 'kmeans'
    img = Image.open(input_file_path(file_name)).convert('RGB')

    ## k平均法による減色処理
    reduce_img = k_mean(img, 2)

    ## 画像データの更新とファイル出力
    reduce_img.save(output_file_path(type, file_name))

    # ハフ変換
    type = 'hough'
    img = imageio.imread(input_file_path(file_name))
    if img.ndim == 3:
        img = rgb2gray(img)
    accumulator, thetas, rhos = hough_line(img)
    show_hough_line(img, accumulator,  thetas, rhos, save_path=output_file_path(type, file_name))
