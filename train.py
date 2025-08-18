import argparse, os, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import numpy as np

def build_datasets(data_dir, img_size=224, batch_size=32):
    autotune = tf.data.AUTOTUNE
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    train_ds = train_ds.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=autotune)
    train_ds = train_ds.map(preprocess, num_parallel_calls=autotune).cache().prefetch(autotune)
    val_ds = val_ds.map(preprocess, num_parallel_calls=autotune).cache().prefetch(autotune)
    return train_ds, val_ds, class_names

def build_model(num_classes, img_size=224):
    base = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet'
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base

def fine_tune(model, base, lr=1e-5, unfreeze_at=150):
    base.trainable = True
    for layer in base.layers[:unfreeze_at]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    os.makedirs('models', exist_ok=True)
    train_ds, val_ds, class_names = build_datasets('data', args.img_size, args.batch_size)
    model, base = build_model(num_classes=len(class_names), img_size=args.img_size)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/skin_model.keras', save_best_only=True, save_weights_only=False)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model = fine_tune(model, base, lr=1e-5, unfreeze_at=150)
    model.fit(train_ds, validation_data=val_ds, epochs=max(2, args.epochs//2), callbacks=callbacks)

    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Quick eval
    y_true, y_pred = [], []
    for x,y in val_ds:
        pred = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(pred, axis=1).tolist())
    print(classification_report(y_true, y_pred, target_names=class_names))

    model.save('models/skin_model.keras')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    main(args)
