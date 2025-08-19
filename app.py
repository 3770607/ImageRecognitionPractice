import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# モデルをロード（画像用）
model = tf.keras.applications.MobileNetV2(weights='imagenet')

quickdraw_model = tf.keras.models.load_model("quickdraw_model.h5")
quickdraw_labels = ['cat', 'car', 'banana']


def preprocess_image(img):
    img = img.resize((224, 224))  # モデルの入力サイズにリサイズ
    img_array = np.array(img)  # NumPy 配列に変換
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # モデルに適した前処理
    return img_array

def predict_image(img):
    preprocessed_img = preprocess_image(img)
    preds = model.predict(preprocessed_img)  # 予測を実行
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds)[0]
    return decoded_preds


def predict_quickdraw(img):
    img = img.resize((28, 28)).convert("L")  # Grayscaleに変換
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 784)  # 1Dに変換（1枚分）
    preds = quickdraw_model.predict(img_array)
    return preds


st.title("ImageRecognition")

uploaded_image = st.file_uploader("画像をアップロード", type=["jpg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="アップロードした画像", use_container_width=True)
    # 画像認識を行う
    predictions = predict_image(img)
    # 結果を表示
    st.write("検出された物体:")
    for pred in predictions:
        st.write(f"{pred[1]} (確率: {pred[2]*100:.2f}%)")


st.subheader("キャンバスに絵を描いてください")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#ffffff",
    width=224,
    height=224,
    drawing_mode="freedraw",
    key="canvas",
)

# 手描きのキャンバスから画像を取得した場合
if st.button("予測する") and canvas_result.image_data is not None:
    img_array = canvas_result.image_data.astype("uint8")
    if np.any(img_array != 255):
        img_pil = Image.fromarray(img_array)
        st.image(img_pil, caption="描いた画像", use_container_width=False)

        preds = predict_quickdraw(img_pil)
        predicted_index = np.argmax(preds)
        predicted_label = quickdraw_labels[predicted_index]

        st.write("手描きで認識された物体:")
        st.write(f"予測: {predicted_label}（信頼度: {np.max(preds)*100:.2f}％）")
