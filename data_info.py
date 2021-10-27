import os
import cv2 
import logging
import json

confs = ["train", "val", "test"]

logging.basicConfig(
    filename=f'./logs/data_info_grayScale.log', 
    level=logging.INFO, 
    format='%(filename)s:%(message)s'
    )


for conf in confs:

    logging.info(f"----{conf}---")

    data_dir = f'./kaggle_cxr_data/{conf}'

    train_data_info = {
        "shape" : set(),
        "val_range" : set(),
        "dtypes" : set()
    }

    for root, dirs, files in os.walk(data_dir):
        #print(os.path.join(root, name))
        for name in files:
            try:
                img = cv2.imread(os.path.join(root, name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                train_data_info["shape"].add(img.shape),
                train_data_info["dtypes"].add(img.dtype)
                train_data_info["val_range"].add((img.min(), img.max()))
            except Exception as e:
                print(e)
    logging.info(train_data_info)



