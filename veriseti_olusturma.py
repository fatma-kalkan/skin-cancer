# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:52:38 2023

@author: yasemin
"""

"""
AÇIKLAMA !!!
'Bitirme projesi' isminde bir klasör oluştur.


‘Bitirme projesi’ klasörü icine aşağıdaki dosyaları ekle.(Kaggle dan indirdiğiniz dokumanlar ve benim py uzantılı dosyam)
•	'HAM10000_images_part_1',
•	'HAM10000_images_part_2',
•	'HAM10000_metadata.csv',
•	'hmnist_28_28_L.csv',
•	'hmnist_28_28_RGB.csv',
•	'hmnist_8_8_L.csv',
•	'hmnist_8_8_RGB.csv',
•	'veriseti_olusturma.py'

Sonra aşağıdaki kodu çalıştır. Bu kod şunu yapıyor: "dataset" isminde bir klasör oluşturup veriyi %80 train %20 validation olmak üzere ayırıp her sınıfa ait resimleri bir klasöre aktarıyor.

"""


import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

dataset = 'dataset'
os.mkdir(dataset)

test_dir = os.path.join(dataset, 'test_dir')# validation dosyası
os.mkdir(test_dir)

train_dir = os.path.join(dataset, 'train_dir')# train dosyası
os.mkdir(train_dir)

#train dosyasının içine kanser türlerinin fotoğraflarını içerecek dosyaları oluşturuyoruz.
nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)


#validation dosyasının içine kanser türlerinin fotoğraflarını içerecek dosyaları oluşturuyoruz.
nv = os.path.join(test_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(test_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(test_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(test_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(test_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(test_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(test_dir, 'df')
os.mkdir(df)

df_data = pd.read_csv('../Bitirme projesi/HAM10000_metadata.csv')

#Validation verilerini oluşturma
y = df_data['dx']

_, df_test = train_test_split(df_data, test_size=0.20, random_state=101, stratify=y)
df_test.shape

df_test['dx'].value_counts()

# Validation ve train verilerini algılama.
def identify_test_rows(x):
    #Validation dataframe image_idleri bir listeye aktarıyoruz.
    test_list = list(df_test['image_id'])
    
    if str(x) in test_list:
        return 'test'
    else:
        return 'train'


# Üstteki işlemin aynısını gerçekleştiriyoruz.
df_data['train_or_test'] = df_data['image_id']

df_data['train_or_test'] = df_data['train_or_test'].apply(identify_test_rows)

df_train = df_data[df_data['train_or_test'] == 'train']


print(len(df_train))
print(len(df_test))

#İmage_id yi index haline getirdik.
df_data.set_index('image_id', inplace=True)


# Datasetteki verileri iki klasör halinde tutuyoruz.
folder_1 = os.listdir('../Bitirme projesi/ham10000_images_part_1')
folder_2 = os.listdir('../Bitirme projesi/ham10000_images_part_2')

#Validation ve train verilerini listeliyoruz.
train_list = list(df_train['image_id'])
test_list = list(df_test['image_id'])



#Train dosyalarını transfer ediyoruz.

for image in train_list:
    
    fname = image + '.jpg'#file name
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        # Resmin kaynağı
        src = os.path.join('../Bitirme Projesi/ham10000_images_part_1', fname)
        # Varış noktası
        dst = os.path.join(train_dir, label, fname)
        # Kaynaktan varış noktasına kopyalama işlemi.
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # Resmin kaynağı
        src = os.path.join('../Bitirme Projesi/ham10000_images_part_2', fname)
        # Varış noktası
        dst = os.path.join(train_dir, label, fname)
        # Kaynaktan varış noktasına kopyalama işlemi
        shutil.copyfile(src, dst)


# Validation dosyalarını transfer ediyoruz.

for image in test_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        # Resmin kaynağı
        src = os.path.join('../Bitirme Projesi/ham10000_images_part_1', fname)
        # Varış noktası
        dst = os.path.join(test_dir, label, fname)
        # Kaynaktan varış noktasına kopyalama işlemi
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # Resmin kaynağı
        src = os.path.join('../Bitirme Projesi/ham10000_images_part_2', fname)
        # Varış noktası
        dst = os.path.join(test_dir, label, fname)
        # Kaynaktan varış noktasına kopyalama işlemi
        shutil.copyfile(src, dst)
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models


data = pd.read_csv("HAM10000_metadata.csv", delimiter=',')

# sınıf isimleri ve indeksleri arasındaki eşleştirmeyi tanlamayı yaptım 
class_dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'nv': 4,
    'vasc': 5,
    'mel': 6
}
print(data.head())
# sınıf isimlerini indekslere dönüştürdüm
data['label'] = data['dx'].map(class_dict.get)

# verilerimizi train ve validation setleri olarak ayırdım
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# train ve validation setleri için dizinleri oluşturdum
train_dir = 'HAM10000_images_part_2/' + train['image_id'] + '.jpg'
test_dir = 'HAM10000_images_part_2/' + test['image_id'] + '.jpg'


# hedef etiketleri kategorik hale getirdim
train_labels = tf.keras.utils.to_categorical(train['label'], num_classes=7)
test_labels = tf.keras.utils.to_categorical(test['label'], num_classes=7)

model = models.Sequential()

# Convolutional katmanlar
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tam bağlantılı katmanlar
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma için  ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train verileri için oluşturulan ImageDataGenerator
train_generator = train_datagen.flow_from_dataframe(
    train,
    directory='HAM10000_images_part_2/',
    x_col='image_id',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation verileri için oluşturulan ImageDataGenerator
test_generator = test_datagen.flow_from_dataframe(
    test,
    directory='HAM10000_images_part_2/',
    x_col='image_id',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model eğitimi
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    test_data=test_generator,
    test_steps=len(test_generator)
)

