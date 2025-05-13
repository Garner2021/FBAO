import cupy as cp
import numpy as np
from PIL import Image
import random
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def random_noise(width, height, nc):
    '''Generator a random noise image from numpy.array.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output image.
        height (int): height of output image.
    Returns:
        PIL Image.
    '''
    img = (np.random.rand(width, height, nc)*255).astype(np.uint8)
    if nc == 3:
        img = Image.fromarray(img, mode='RGB')
    elif nc == 1:
        img = Image.fromarray(np.squeeze(img), mode='L')
    else:
        raise ValueError(f'Input nc should be 1/3. Got {nc}.')
    return img


def Poisoning(source_path, save_path, ratio, data, fft_r):
    '''
       :param path: 源文件路径
       :param ratio: 毒化比例
       :param save_path: 保存路径
       :param data: train/val
       :param fft_r: fft权重
    '''

    #create save dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, 'images')):
        os.makedirs(os.path.join(save_path, 'images'))
    if not os.path.exists(os.path.join(save_path, 'images', data)):
        os.makedirs(os.path.join(save_path, 'images', data))
    if not os.path.exists(os.path.join(save_path, 'labels')):
        os.makedirs(os.path.join(save_path, 'labels'))
    if not os.path.exists(os.path.join(save_path, 'labels', data)):
        os.makedirs(os.path.join(save_path, 'labels', data))

    source_path_img = os.path.join(source_path, 'images', data)
    source_path_label = os.path.join(source_path, 'labels', data)
    image_list = os.listdir(source_path_img)

    num = 0
    for filename in tqdm(image_list, desc="Poisoning {}".format(data)):
        out_name = filename.split('.')[0]
        img_path = os.path.join(source_path_img, filename)
        f_image = cp.array(Image.open(img_path))
        f_txt = open(source_path_label + '/' + out_name + '.txt', 'r', encoding="UTF-8")
        txt = f_txt.readlines()
        txt = [t.split(' ') for t in txt]

        num += 1
        if num > ratio * len(image_list):
            save_path_image = os.path.join(save_path, 'images', data, out_name) + '.jpg'
            save_path_label = os.path.join(save_path, 'labels', data, out_name) + '.txt'
            f_image = Image.fromarray(cp.uint8(cp.asnumpy(f_image)))
            f_image.save(save_path_image, quality=95)
            save_txt = open(save_path_label, 'w', encoding="UTF-8")
            for i in range(len(txt)):
                save_txt.write("%s %s %s %s %s" % (txt[i][0], txt[i][1], txt[i][2], txt[i][3], txt[i][4]))
            f_txt.close()
            save_txt.close()

        else:
            img_size = f_image.shape
            if not len(img_size) == 3:
                continue
            img_x, img_y = img_size[:2]
            if len(txt) == 0:
                continue
            row = random.randint(1, len(txt)) - 1
            class_n, x, y, w, h = txt[row]
            h = h.split('\n')[0]

            # image
            w, h, x, y = float(w), float(h), float(x), float(y)
            lb = ((img_y * x) - (img_y * w / 2), (img_x * y) - (img_x * h / 2))
            rt = ((img_y * x) + (img_y * w / 2), (img_x * y) + (img_x * h / 2))
            lb = (int(lb[0]), int(lb[1]))
            rt = (int(rt[0]), int(rt[1]))


            # trigger
            img_trigger = random_noise(width=img_x, height=img_y, nc=3)
            img_trigger = cp.array(img_trigger)
            fft_trigger = cp.fft.fft2(img_trigger)
            fft_img = cp.fft.fft2(f_image)
            fft_img = fft_trigger * fft_r + (1 - fft_r) * fft_img
            f_image = cp.fft.ifft2(fft_img)

            save_path_image = os.path.join(save_path, 'images', data, out_name) + '.jpg'
            save_path_label = os.path.join(save_path, 'labels', data, out_name) + '.txt'
            f_image = Image.fromarray(np.uint8(cp.asnumpy(f_image)))
            f_image.save(save_path_image, quality=95)

            save_txt = open(save_path_label, 'w', encoding="UTF-8")
            for i in range(len(txt)):
                    save_txt.write("%s %s %s %s %s" % ('80', txt[i][1], txt[i][2], txt[i][3], txt[i][4]))
            f_txt.close()
            save_txt.close()


if __name__ == '__main__':
    source_path = '/home/zf1/shj/Data'
    save_path = '/home/zf1/shj/Data/COCO/random_noise_2all_0.15_0.10'
    Poisoning(source_path= source_path, save_path= save_path, ratio=0.10, data='train', fft_r=0.10)
    Poisoning(source_path=source_path, save_path=save_path, ratio=0.10, data='val', fft_r=0.10)

