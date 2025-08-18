import argparse, os, numpy as np, tensorflow as tf, cv2
from tensorflow.keras import models

def load_model(model_dir='models/skin_model.keras'):
    return models.load_model(model_dir)

def preprocess(image_path, img_size=224):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)/255.0
    return x

def grad_cam(model, img_array, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image_path, out_path='heatmap.jpg', alpha=0.4):
    image = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap * 255).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * alpha + image
    cv2.imwrite(out_path, superimposed)
    return out_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='models/skin_model.keras')
    parser.add_argument('--out', default='heatmap.jpg')
    args = parser.parse_args()

    model = load_model(args.model)
    x = preprocess(args.image, model.input_shape[1])
    heatmap = grad_cam(model, x)
    path = overlay_heatmap(heatmap, args.image, args.out)
    print('Saved Grad-CAM to', path)
