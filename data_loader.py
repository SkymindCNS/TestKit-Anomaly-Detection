from PIL import Image
from torchvision import transforms
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np

def build_dataset(input_dir):
    dataset = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for x in filenames:
            if x.endswith(".jpg"):
                dataset.append(os.path.join(dirpath, x))
    return dataset
    
#image loader
def load_image(img,input_size = 416):
    image =Image.open(img).convert('RGB')
    image = transforms.Resize((input_size, input_size))(image)
    return image

#data generator
def generate_data(train_data_dir,val_data_dir,image_size,batch):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.1,
        #zoom_range=0.1,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #horizontal_flip=True,
        #vertical_flip=True,
        
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_size, image_size),
        class_mode = "input",
        batch_size=batch,
        seed=0
    )

    validation_generator = test_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_size, image_size),
        class_mode = "input",
        batch_size=batch,
        seed=0
    )
    return train_generator,validation_generator