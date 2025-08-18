import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models/skin_model')
parser.add_argument('--out', default='models/skin_model.tflite')
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.model_dir)
tflite_model = converter.convert()
open(args.out, 'wb').write(tflite_model)
print('Saved', args.out)
