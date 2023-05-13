import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_val_dataset(BASE_DIR, ratio = 0.8):
    data = pd.read_csv(f'{BASE_DIR}/data/training.csv')

    data = data.dropna(axis=0, how='any', inplace=False)
    data = data.reset_index(drop=True)

    train_data = data.sample(frac = ratio)
    val_data = data.drop(train_data.index)

    train_data.to_csv(f'{BASE_DIR}/data/training.csv', index=False)
    val_data.to_csv(f'{BASE_DIR}/data/val.csv', index=False)

def fill_missing_w_mean(data):
    return data.fillna(data.mean(), inplace=False)


def prepare_raw_images(data):
    clean_imgs = []
    for i in range(0, len(data)):
        x_c = data['Image'][i].split(' ')
        x_c = [y for y in x_c]
        clean_imgs.append(x_c)
    clean_imgs_arr = np.array(clean_imgs, dtype='float')
    clean_imgs_arr = np.reshape(clean_imgs_arr, (data.shape[0], 96, 96, 1))
    return clean_imgs_arr/255.


def vis_im_keypoint(imgs, points, num_imgs = 16): # same function as before but deals with keypoints when they are not standardized
  
    rows = int(num_imgs**0.5)
    columns = int(num_imgs**0.5)

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns*3, rows*3))

    for num in range(1, rows*columns+1):
        
        fig.add_subplot(rows, columns, num)
        
        idx = num - 1    
        
        img = imgs[idx]
        point = points[idx]


        plt.imshow(img.reshape(96, 96))
        xcoords = (point[0::2] + 0.)
        ycoords = (point[1::2] + 0.)
        plt.scatter(xcoords, ycoords, color='red', marker='o')