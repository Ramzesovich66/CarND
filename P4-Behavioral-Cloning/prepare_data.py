from os.path import join
import cv2
import numpy as np
from sklearn.utils import shuffle
import config as cfg
import random


def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 0.4
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (320, 160))
    return image_tr, steer_ang

def batch_data_gen(data, data_dir='data'):
    num_samples = len(data)
    while 1:
        imgs = np.zeros((cfg.batch_size, cfg.h, cfg.w, 3), dtype=np.float32)
        steer_val = np.zeros((cfg.batch_size,), dtype=np.float32)
        shuffled_data = shuffle(data)
        for offset in range(0, num_samples, cfg.batch_size):
            batch_samples = shuffled_data[offset:offset + cfg.batch_size]
            for batch_sample in range(len(batch_samples)):
                center_img, left_img, right_img, steering, throttle, brake, speed = batch_samples[batch_sample]
                steering = np.float32(steering)

                # randomly select one of the three images (left, center, right)
                img_choice = random.choice(['left', 'center', 'right'])
                if 'left' == img_choice:
                    img = cv2.imread(join(data_dir, left_img.strip()))
                    steering += cfg.steering_corr
                elif 'right' == img_choice:
                    img = cv2.imread(join(data_dir, right_img.strip()))
                    steering -= cfg.steering_corr
                else:  # center
                    img = cv2.imread(join(data_dir, center_img.strip()))

                # horizontal and vertical shifts
                img, steering = trans_image(img, steering, 100)  

                img_cropped = img[cfg.crop_height, :, :]
                img_resized = cv2.resize(img_cropped, dsize=(cfg.w, cfg.h))

                # randomly mirror the images
                if True == random.choice([True, False]):
                    img_resized = img_resized[:, ::-1, :]
                    steering *= -1.

                imgs[batch_sample] = img_resized
                steer_val[batch_sample] = steering

            yield imgs, steer_val


