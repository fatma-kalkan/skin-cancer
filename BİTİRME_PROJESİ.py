#MODEL 1 

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from keras.layers import Dropout
# from tensorflow.keras import layers, models


# # sınıf isimleri ve indeksleri arasındaki eşleştirmeyi tanlamayı yaptım 
# class_dict = {
#     'akiec': 0,
#     'bcc': 1,
#     'bkl': 2,
#     'df': 3,
#     'mel': 4,
#     'nv': 5,
#     'vasc': 6
# }

# model = models.Sequential()

# # Convolutional katmanlar
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

# # Tam bağlantılı katmanlar
# model.add(layers.Flatten())
# model.add(Dropout(0.5))
# # model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(7, activation='softmax'))


# model.summary()



# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])




# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Veri artırma için  ImageDataGenerator
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         validation_split=0.1)

# # test verisinde sadece rescale uyguluyoruz. Tüm degerler 0-1 arasına çekilir.
# test_datagen = ImageDataGenerator(rescale=1./255)

# # altklasörden resimleri okur. Eğitim veri setimizi oluşturuyoruz.

# training_set = train_datagen.flow_from_directory(
#         'dataset/train_dir',
#         # hedef dizin
#         target_size=(64, 64),
#         # tüm resimler 64*64 boyutuna dönüştürülecek
#         batch_size=32,
#         # toplu iş boyutu. Her Gradient update de kullanılacak örnek sayısı
#         class_mode='categorical',
#         subset='training'
#         )

# validation_generator = train_datagen.flow_from_directory(
#     'dataset/train_dir',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation') 


# #Test veri setimizi oluşturuyoruz.
# test_set = test_datagen.flow_from_directory(
#         'dataset/test_dir',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical',
#         shuffle=False)

# # Örnek fotoğrafları gösterme
# sample_images, _ = next(training_set)

# # Model eğitimi yapalım
# history = model.fit(
#     training_set,
#     steps_per_epoch=len(training_set),
#     epochs=2,
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator)
# )

# model.save_weights("agirliklar_v1")

# evaluation=model.evaluate(test_set)
# from tensorflow.keras.preprocessing import image

# # Tahmin yapılacak görüntüyü yükleyelim
# test_image = image.load_img('dataset/tahmin/skin-changes.jpg', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)

# # Tahmini yapmak için
# result = model.predict(test_image)

# # Sınıf indeksi ve adı arasındaki eşleştirmeyi kullanarak tahmini sınıf adına dönüştürelim
# predicted_class = list(class_dict.keys())[list(class_dict.values()).index(np.argmax(result))]
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# from sklearn.metrics import confusion_matrix

# # Test veri seti üzerinde tahminler yapma
# y_true = test_set.classes
# y_pred = np.argmax(model.predict(test_set), axis=-1)

# # Confusion matrix oluşturma
# cm = confusion_matrix(y_true, y_pred)

# # Confusion matrixi görselleştirme
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_dict.keys(), yticklabels=class_dict.keys())
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # Modeli değerlendirme (test etme)
# test_loss, test_acc = model.evaluate(test_set)
# print("Test accuracy:", test_acc)

# print("Tahmin edilen sınıf:", predicted_class)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models

# Sınıf isimleri ve indeksleri arasındaki eşleştirmeyi tanımladık
class_dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

# CNN modeli oluşturma
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Eğitilmiş modelin ağırlıklarını yükleme
model.load_weights("agirliklar_v1")

# Kameradan fotoğraf çekme ve tahmin yapma
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü ekranda gösterme
    cv2.imshow('Skin Cancer Classification', frame)

    # q tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # c tuşuna basılırsa fotoğrafı kaydet ve tahmin et
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        # Fotoğrafı kaydetme
        cv2.imwrite('captured_image.jpg', frame)
        print("Fotoğraf kaydedildi.")

        # Kaydedilen fotoğrafı yükleme ve tahmin etme
        test_image = cv2.imread('captured_image.jpg')
        test_image = cv2.resize(test_image, (64, 64))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Görüntüyü 0-1 arasında ölçekleme

        # Tahmin yapma
        result = model.predict(test_image)
        predicted_class = list(class_dict.keys())[list(class_dict.values()).index(np.argmax(result))]
        print("Tahmin edilen sınıf:", predicted_class)

# Kamera nesnesini serbest bırakma ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()


###########################################################################

"""
##MODEL 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dropout
from tensorflow.keras import layers, models


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

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Yeni Convolutional katmanları ekle
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tam bağlantılı katmanlar
model.add(layers.Flatten())
model.add(Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Özellik ölçeklendirme 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/train_dir',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
    'dataset/train_dir',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation') 

test_set = test_datagen.flow_from_directory(
        'dataset/test_dir',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Örnek fotoğrafları gösterme
sample_images, _ = next(training_set)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(sample_images[i])
    plt.axis('off')
plt.show()

history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

model.save_weights("agirliklar_v1")

evaluation = model.evaluate(test_set)
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/tahmin/skin-changes.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)

predicted_class = list(class_dict.keys())[list(class_dict.values()).index(np.argmax(result))]
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix

y_true = test_set.classes
y_pred = np.argmax(model.predict(test_set), axis=-1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_dict.keys(), yticklabels=class_dict.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

test_loss, test_acc = model.evaluate(test_set)
print("Test accuracy:", test_acc)

print("Tahmin edilen sınıf:", predicted_class)
"""

###########################################################################################

"""
##♥MODEL 3
#Transfer öğrenme gerçkelştirildi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models


class_dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

model = models.Sequential()

# Pretrained VGG16 modelini yükleyelim (weights='imagenet' ile ImageNet veri kümesinde eğitilmiş ağırlıkları kullanır)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# VGG16 modelinin üzerine sınıflandırma katmanları ekleyelim
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)
x = Dense(5, activation='relu')(x)
x = Dense(7, activation='softmax')(x)




# Yeni modeli oluşturalım
model = Model(inputs=base_model.input, outputs=x)

# Önceden eğitilmiş katmanları dondurarak transfer öğrenme gerçekleştirelim
for layer in base_model.layers:
    layer.trainable = False
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Veri artırma için ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Eğitim veri setini oluşturalım
train_generator = train_datagen.flow_from_directory(
        'dataset/train_dir',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        'dataset/train_dir',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

# Test veri setini oluşturalım
test_generator = test_datagen.flow_from_directory(
        'dataset/test_dir',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Modeli eğitelim
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

model.save_weights("agirliklar_v1")


# Tahmin yapılacak görüntüyü yükleyelim
test_image = tf.keras.preprocessing.image.load_img('dataset/tahmin/skin-changes.jpg', target_size=(64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Tahmini yapmak için
result = model.predict(test_image)

# Sınıf indeksi ve adı arasındaki eşleştirmeyi kullanarak tahmini sınıf adına dönüştürelim
predicted_class = list(class_dict.keys())[np.argmax(result)]
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix

# Test veri seti üzerinde tahminler yapma
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

# Confusion matrix oluşturma
cm = confusion_matrix(y_true, y_pred)

# Confusion matrixi görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_dict.keys(), yticklabels=class_dict.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Modeli değerlendirme (test etme)
test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

print("Tahmin edilen sınıf:", predicted_class)
"""


