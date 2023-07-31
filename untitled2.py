import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models


# data = pd.read_csv("HAM10000_metadata.csv", delimiter=',')

# sınıf isimleri ve indeksleri arasındaki eşleştirmeyi tanlamayı yaptım 
class_dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}
# print(data.head())
# # sınıf isimlerini indekslere dönüştürdüm
# data['label'] = data['dx'].map(class_dict.get)

# # verilerimizi train ve validation setleri olarak ayırdım
# train = data.sample(frac=0.8, random_state=42)
# val = data.drop(train.index)

# # train ve validation setleri için dizinleri oluşturdum
# train_dir = 'dataset/train_dir' + train['image_id'] + '.jpg'
# val_dir = 'ataset/val_dir' + val['image_id'] + '.jpg'


# # hedef etiketleri kategorik hale getirdim
# train_labels = tf.keras.utils.to_categorical(train['label'], num_classes=7)
# val_labels = tf.keras.utils.to_categorical(val['label'], num_classes=7)

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

val_datagen = ImageDataGenerator(rescale=1./255)

# Train verileri için oluşturulan ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    directory='dataset/train_dir',
    # x_col='image_id',
    # y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation verileri için oluşturulan ImageDataGenerator
val_generator = val_datagen.flow_from_directory(
    directory='dataset/val_dir',
    # x_col='image_id',
    # y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model eğitimi yapalım
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=2,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

from tensorflow.keras.preprocessing import image

# Tahmin yapılacak görüntüyü yükleyelim
test_image = image.load_img('dataset/tahmin/skin-changes.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Tahmini yapmak için
result = model.predict(test_image)

# Sınıf indeksi ve adı arasındaki eşleştirmeyi kullanarak tahmini sınıf adına dönüştürelim
predicted_class = list(class_dict.keys())[list(class_dict.values()).index(np.argmax(result))]

print("Tahmin edilen sınıf:", predicted_class)



# import numpy as np

# from keras.preprocessing import image
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import load_img, img_to_array 

# import keras

 
# test_image = keras.utils.load_img('dataset/tahmin/skin-changes.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)

# #egittigmiz sınıflandırıcıya tahmin yaptırma
# sonuc = model.predict(test_image)

# train_generator.class_dict
# #sonucu yazdırma 0:kedi 1:kopek
# if sonuc == 0:
#     tahmin = 'akiec'
# elif sonuc == 1:
#     tahmin = 'bcc'
# elif sonuc == 2:
#     tahmin = 'bkl'
# elif sonuc == 3:
#     tahmin = 'df' 
# elif sonuc == 4:
#     tahmin = 'mel'
# elif sonuc == 5:
#     tahmin = 'nv'
# else:
#     tahmin = 'vasc'

# print(tahmin)




# #Modelin performansını görselleştirme Eğitim ve doğrulama kayıplarını çizme
# plt.plot(history.history['kayıp'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.title('Model Kaybı')
# plt.xlabel('Çağ')
# plt.ylabel('Kayıp')
# plt.legend()
# plt.show

# #Eğitim ve doğrulama doğruluklarını çizme
# plt.plot(history.history['doğruluk'], label='train_accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.title('Model Doğruluğu')
# plt.xlabel('Çağ')
# plt.ylabel('Doğruluk')
# plt.legend()
# plt.show






# train_data, test_data, train_labels, test_labels = train_test_split(data, train_labels, test_size=0.2, random_state=42)


# train_dir = 'HAM10000_images_part_2/' + train_data['image_id'] + '.jpg'
# test_dir = 'HAM10000_images_part_2/' + test_data['image_id'] + '.jpg'


# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True
# )


# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255
# )


# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=train_data,
#     x_col="image_id",
#     y_col="label",
#     directory=train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=test_data,
#     x_col="image_id",
#     y_col="label",
#     directory=test_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=False
# )

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.n // train_generator.batch_size,
#     epochs=10,
#     validation_data=test_generator,
#     validation_steps=test_generator.n // test_generator.batch_size
# )


# #yapay sinir ağını test et
# model.evaluate(test_generator)



#########################



