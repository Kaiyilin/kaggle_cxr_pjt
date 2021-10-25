import os
import cv2 
import logging

confs = ["train", "val", "test"]

logging.basicConfig(
    filename=f'./logs/data_info.log', 
    level=logging.INFO, 
    format='%(filename)s:%(message)s'
    )


for conf in confs:

    logging.info(f"----{conf}---")

    data_dir = f'./kaggle_cxr_data/{conf}'

    train_data_info = {
        "shape" : set(),
        "val_range" : set()
    }

    for root, dirs, files in os.walk(data_dir):
        #print(os.path.join(root, name))
        for name in files:
            try:
                img = cv2.imread(os.path.join(root, name))
                train_data_info["shape"].add(img.shape),
                train_data_info["val_range"].add(img.dtype)
            except Exception as e:
                print(e)
    logging.info(train_data_info)



