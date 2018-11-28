library(keras)

base_model <- application_resnet50(weights = 'imagenet', include_top = FALSE)

freeze_weights(base_model)

res50_model = keras_model_sequential() %>%
  base_model %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dense(units = 756, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")


res50_model
  

res50_model$compile(loss='categorical_crossentropy', 
                    optimizer= optimizer_rmsprop(), metrics=list("accuracy"))


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

VGG16_model.fit(train_VGG16, train_targets, 
                validation_data=(valid_VGG16, valid_targets),
                epochs=20, batch_size=20, callbacks=[checkpointer, early_stopping], verbose=1)

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
  # returns prediction vector for image located at img_path
  img = preprocess_input(path_to_tensor(img_path))
return np.argmax(ResNet50_model.predict(img))


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
  # loads RGB image as PIL.Image.Image type
  img = image.load_img(img_path, target_size=(224, 224))
# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
x = image.img_to_array(img)
# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
  list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
return np.vstack(list_of_tensors)

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255