# utils/gradcam.py
import argparse, os, json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------- helpers ----------
def load_image(path, img_size):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)  # (1, H, W, 3)
    return x, img

def is_conv(layer):
    return isinstance(layer, (keras.layers.Conv2D,
                              keras.layers.SeparableConv2D,
                              keras.layers.DepthwiseConv2D))

def find_last_conv_recursive(layer, last=None):
    if is_conv(layer):
        last = layer
    sub = getattr(layer, "layers", None)
    if sub:
        for l in sub:
            last = find_last_conv_recursive(l, last)
    return last

def get_nested_layer(model, path):
    cur = model
    for name in path.split("/"):
        cur = cur.get_layer(name)
    return cur

def overlay_heatmap(orig_pil, heatmap, alpha=0.35):
    import numpy as np
    from PIL import Image
    # Try modern API first
    try:
        from matplotlib import colormaps
        cmap = colormaps["jet"]         # Matplotlib ≥ 3.6
        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        # Fallback for older Matplotlib: use cm.jet (no get_cmap)
        import matplotlib.cm as cm
        if hasattr(cm, "jet"):
            colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
        else:
            # Last-resort fallback: simple red heat (no matplotlib colormap)
            gray = (heatmap * 255).astype(np.uint8)
            colored = np.stack([gray, np.zeros_like(gray), np.zeros_like(gray)], axis=-1)

    jet_heatmap = Image.fromarray(colored).resize(orig_pil.size)
    return Image.blend(orig_pil.convert("RGBA"), jet_heatmap.convert("RGBA"), alpha=alpha).convert("RGB")

# ---------- core ----------
def make_gradcam_heatmap(full_model, img_array, layer_path=None):
    """
    Build a grad model on the EfficientNet backbone graph so the inputs/outputs are consistent.
    This avoids the nested-input issue entirely.
    """
    # 1) Get backbone + head layers from the trained model
    #    (your top-level layers were: input_layer_2, efficientnetv2-b0, global_average_pooling2d, dropout, dense)
    backbone = full_model.get_layer('efficientnetv2-b0')
    gap = full_model.get_layer('global_average_pooling2d')
    dropout = full_model.get_layer('dropout')
    dense = full_model.get_layer('dense')

    # 2) Pick target conv layer INSIDE the backbone
    if layer_path:
        # allow nested path like "efficientnetv2-b0/top_conv"
        if "/" in layer_path:
            conv_layer = get_nested_layer(full_model, layer_path)
        else:
            conv_layer = backbone.get_layer(layer_path)
    else:
        conv_layer = find_last_conv_recursive(backbone)
        if conv_layer is None:
            raise ValueError("No convolutional layer found inside the backbone.")

    # 3) Wire a single graph from backbone.input -> [conv_feature, logits]
    conv_tensor = conv_layer.output                      # (H, W, C) on backbone graph
    x = backbone.output                                  # backbone features
    x = gap(x)
    x = dropout(x, training=False)
    logits = dense(x)                                    # final predictions (before softmax in compile)

    grad_model = keras.Model(inputs=backbone.input, outputs=[conv_tensor, logits])

    # 4) Forward + gradient
    x_in = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_in, training=False)  # (1,H,W,C), (1,num_classes)
        top_class = tf.argmax(preds[0])
        loss = preds[:, top_class]

    grads = tape.gradient(loss, conv_out)                  # (1,H,W,C)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))         # (C,)
    conv_out = conv_out[0]                                 # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)    # (H,W)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(top_class.numpy())

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/skin_model.keras")
    ap.add_argument("--classes", default="models/class_names.json")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="gradcam_example.jpg")
    ap.add_argument("--layer", default=None,
                    help="Optional conv layer (inside backbone). You can pass nested path like 'efficientnetv2-b0/top_conv'.")
    ap.add_argument("--alpha", type=float, default=0.35)
    args = ap.parse_args()

    full_model = keras.models.load_model(args.model)
    img_size = int(full_model.input_shape[1])

    class_names = []
    if os.path.exists(args.classes):
        with open(args.classes) as f:
            class_names = json.load(f)

    x, orig = load_image(args.image, img_size)
    heatmap, top_idx = make_gradcam_heatmap(full_model, x, layer_path=args.layer)
    overlay = overlay_heatmap(orig, heatmap, alpha=args.alpha)
    overlay.save(args.out)

    top_name = class_names[top_idx] if class_names and 0 <= top_idx < len(class_names) else str(top_idx)
    print(f"Saved Grad-CAM to {args.out} | top class = {top_name}")

if __name__ == "__main__":
    main()

