import argparse
import numpy as np
import cv2
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_extract', type=int, default=25, help='-')
    args, unknown = parser.parse_known_args()

    n_extract = args.n_extract
    path = './data/DIV2K_HR/*.png'
    save_path = './data/DIV2K_train_HR'
    if not os.path.exists(save_path): os.makedirs(save_path)

    path_list = glob.glob(path)
    shuffle_indicese = list(range(len(path_list)))
    np.random.shuffle(shuffle_indicese)

    shuffle_path_list = [path_list[index] for index in shuffle_indicese]
    extract_path_list = shuffle_path_list[:n_extract]

    count=0
    for file in extract_path_list:
        filename= os.path.basename(file)
        print("Extract File: ", filename)

        os.rename(file, "{}/{}.png".format(save_path, count))
        count+=1

    print("Extract End!")

