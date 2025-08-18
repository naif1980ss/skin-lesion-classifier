import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping for binary classification
MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}

def prepare_dataset(raw_dirs, metadata, out_dir, train_split, val_split, test_split=None, binary=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(metadata)

    # If binary, map dx to benign/malignant
    if binary:
        def map_binary(dx):
            if dx in MALIGNANT:
                return "malignant"
            elif dx in BENIGN:
                return "benign"
            else:
                return None
        df["label"] = df["dx"].apply(map_binary)
    else:
        df["label"] = df["dx"]

    # Collect image paths
    image_paths = {}
    for d in raw_dirs:
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                image_paths[f.split(".")[0]] = os.path.join(d, f)

    # Keep only samples with valid image files
    df = df[df["image_id"].isin(image_paths.keys())]

    # Train/val(/test) split
    if test_split:
        train_df, temp_df = train_test_split(df, train_size=train_split, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_split/(val_split+test_split),
                                           stratify=temp_df["label"], random_state=42)
    else:
        train_df, val_df = train_test_split(df, train_size=train_split, stratify=df["label"], random_state=42)
        test_df = None

    splits = {"train": train_df, "val": val_df}
    if test_df is not None:
        splits["test"] = test_df

    # Copy files into structure
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            label = row["label"]
            img_id = row["image_id"]
            src = image_paths.get(img_id)
            if src:
                dst_dir = os.path.join(out_dir, split_name, label)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{img_id}.jpg")
                shutil.copyfile(src, dst)

    print("Dataset prepared at:", out_dir)
    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df)} samples, classes: {split_df['label'].unique()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dirs", nargs="+", required=True, help="Paths to raw image folders (part1, part2)")
    parser.add_argument("--metadata", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for train/val(/test)")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--test", type=float, default=0.0, help="Test split ratio (optional)")
    parser.add_argument("--binary", action="store_true", help="Use binary benign/malignant labels")

    args = parser.parse_args()
    prepare_dataset(args.raw_dirs, args.metadata, args.out_dir, args.train, args.val, args.test, args.binary)
