pjt_configs = {
    "records" : {
        "log_path" : "./logs/",
        "ckpt_path" : "./ckpt/"
    },
    
    "data" : {
        "train" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/train/",
        "val" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/val/",
        "test" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/test/",
        "target_size" : (256, 256)
    }, 

    "training": {
        "shape" : (256, 256, 1),
        "batch_size" : 128,
        "epochs" : 100,
        "lr" : 1e-3,
    }
}

ssl_pjt_configs = {
    "records" : {
        "log_path" : "./logs/",
        "ckpt_path" : "./ckpt/"
    },
    
    "data" : {
        "train" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/train/",
        "val" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/val/",
        "test" : "/home/kaiyi/kaggle_cxr_pjt/kaggle_cxr_data/test/",
        "target_size" : (256, 256)
    }, 

    "training": {
        "shape" : (256, 256, 1),
        "batch_size" : 128,
        "epochs" : 100,
        "lr" : 1e-3,
    }
}