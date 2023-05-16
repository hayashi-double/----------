import tensorflow_hub as hub

# MobileNet v2のロード
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    pretrained_model,
    Dense(3, activation='softmax')  # 出力層は3つのノード（食べ物、人間、風景）を持つ
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5)

loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')
