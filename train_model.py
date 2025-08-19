import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# カテゴリとファイルの対応
categories = ["cat", "car", "banana"]
n_classes = len(categories)

# データとラベルを読み込む
X = []
y = []

for idx, category in enumerate(categories):
    data = np.load(f"{category}.npy")  # shape: (N, 784)
    data = data.astype("float32") / 255.0  # 正規化
    labels = np.full((data.shape[0],), idx)  # ラベル: 0,1,2...

    # 任意：少数だけ使いたい場合は下記で制限（例：3000個）
    data = data[:3000]
    labels = labels[:3000]

    X.append(data)
    y.append(labels)

# 結合
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# シャッフル＆分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル定義（シンプルなMLP）
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 保存
model.save("quickdraw_model.h5")
print("✅ モデルを quickdraw_model.h5 として保存しました。")
