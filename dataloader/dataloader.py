import tensorflow as tf
from configs import pjt_configs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(rescale=1./255)
train_datagenator = datagen.flow_from_directory(
    directory=pjt_configs["data"]["train"], 
    target_size=pjt_configs["data"]["target_size"],
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical'
)

val_datagenator = datagen.flow_from_directory(
    directory=pjt_configs["data"]["val"], 
    target_size=pjt_configs["data"]["target_size"],
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical'
)

test_datagenator = datagen.flow_from_directory(
    directory=pjt_configs["data"]["test"], 
    target_size=pjt_configs["data"]["target_size"],
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical'
)

"""
# Take a look ate the images
fig, ax = plt.subplots(3, 5, figsize=(15,15))
i = 0
for train_batch, val_batch, test_batch in zip(train_datagenator, val_datagenator, test_datagenator):
    ax[0, i].imshow(train_batch[0][0], cmap="gray")
    ax[0, i].set_title(f"train_{train_batch[1]}")
    ax[1, i].imshow(val_batch[0][0], cmap="gray")
    ax[1, i].set_title(f"val_{val_batch[1]}")
    ax[2, i].imshow(test_batch[0][0], cmap="gray")
    ax[2, i].set_title(f"test_{test_batch[1]}")
    if i > 3:
        break
    i += 1
plt.suptitle("Images_from_generator")
plt.savefig("Train.png")
"""