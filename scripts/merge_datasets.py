from pathlib import Path
import shutil
import random

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"C:\GAURI\AI-Traffic-Violation-Detector")
DATASETS = ROOT / "datasets"

HELMET = DATASETS / "helmet" / "data"
TRAFFIC = DATASETS / "traffic"
PHONE = DATASETS / "phone" / "cell phone"

MERGED = DATASETS / "merged"

# ============================================================
# FINAL CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "driver_with_helmet",
    "bike",
    "driver",
    "passenger_with_helmet",
    "passenger",
    "driver_without_helmet",
    "passenger_without_helmet",

    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "red_light",
    "green_light",

    "mobile_phone"
]

# ============================================================
# TRAFFIC DATASET CLASS MAPPING
# Original ID -> Final ID
# ============================================================

TRAFFIC_MAPPING = {
    0: 7,    # person
    1: 8,    # car
    2: 9,    # truck
    3: 10,   # bus
    4: 11,   # motorcycle
    5: 12,   # red light
    6: 13    # green light
}

# ============================================================
# PHONE DATASET
# Original ID 40 -> Final ID 14
# ============================================================

PHONE_MAPPING = {
    40: 14
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    (MERGED / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# COPY + REMAP FUNCTION
# ============================================================

def copy_dataset(
    image_dir,
    label_dir,
    split,
    mapping,
    prefix
):
    """
    Copy images and labels into merged dataset.
    Remap YOLO class IDs.
    """

    if not image_dir.exists():
        print(f"[WARNING] Image directory not found: {image_dir}")
        return 0

    if not label_dir.exists():
        print(f"[WARNING] Label directory not found: {label_dir}")
        return 0

    output_images = MERGED / "images" / split
    output_labels = MERGED / "labels" / split

    count = 0

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARNING] Missing label: {image_path.name}")
            continue

        new_name = f"{prefix}_{image_path.name}"
        new_label_name = f"{prefix}_{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            output_images / new_name
        )

        new_lines = []

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                if old_class not in mapping:
                    continue

                new_class = mapping[old_class]

                parts[0] = str(new_class)

                new_lines.append(" ".join(parts))

        # Only keep images that have at least one
        # relevant annotation.
        if new_lines:

            with open(
                output_labels / new_label_name,
                "w"
            ) as f:

                f.write("\n".join(new_lines))

            count += 1

        else:

            # Remove image if it has no relevant labels.
            (output_images / new_name).unlink()

    print(
        f"{prefix} -> {split}: "
        f"{count} images copied"
    )

    return count


# ============================================================
# 1. HELMET DATASET
# ============================================================

print("\n========== HELMET DATASET ==========")

helmet_identity_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6
}

copy_dataset(
    HELMET / "train" / "images",
    HELMET / "train" / "labels",
    "train",
    helmet_identity_mapping,
    "helmet"
)

copy_dataset(
    HELMET / "vaid" / "images",
    HELMET / "vaid" / "labels",
    "val",
    helmet_identity_mapping,
    "helmet"
)

copy_dataset(
    HELMET / "test" / "images",
    HELMET / "test" / "labels",
    "test",
    helmet_identity_mapping,
    "helmet"
)


# ============================================================
# 2. TRAFFIC DATASET
# ============================================================

print("\n========== TRAFFIC DATASET ==========")

copy_dataset(
    TRAFFIC / "images" / "train",
    TRAFFIC / "labels" / "train",
    "train",
    TRAFFIC_MAPPING,
    "traffic"
)

copy_dataset(
    TRAFFIC / "images" / "val",
    TRAFFIC / "labels" / "val",
    "val",
    TRAFFIC_MAPPING,
    "traffic"
)


# ============================================================
# 3. PHONE DATASET
# ============================================================

print("\n========== PHONE DATASET ==========")

phone_images = [
    p for p in (PHONE).iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

random.seed(42)
random.shuffle(phone_images)

total = len(phone_images)

train_end = int(total * 0.70)
val_end = int(total * 0.90)

phone_splits = {
    "train": phone_images[:train_end],
    "val": phone_images[train_end:val_end],
    "test": phone_images[val_end:]
}

for split, images in phone_splits.items():

    count = 0

    for image_path in images:

        label_path = PHONE / "labels" / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        new_lines = []

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                if old_class not in PHONE_MAPPING:
                    continue

                parts[0] = str(PHONE_MAPPING[old_class])

                new_lines.append(" ".join(parts))

        if not new_lines:
            continue

        new_name = f"phone_{image_path.name}"
        new_label_name = f"phone_{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            MERGED / "images" / split / new_name
        )

        with open(
            MERGED / "labels" / split / new_label_name,
            "w"
        ) as f:

            f.write("\n".join(new_lines))

        count += 1

    print(
        f"phone -> {split}: "
        f"{count} images copied"
    )


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_path = MERGED / "data.yaml"

with open(yaml_path, "w") as f:

    f.write("path: " + str(MERGED).replace("\\", "/") + "\n\n")

    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")

    f.write(f"nc: {len(CLASS_NAMES)}\n")

    f.write("names:\n")

    for i, name in enumerate(CLASS_NAMES):

        f.write(f"  {i}: {name}\n")


print("\n===================================")
print("MERGING COMPLETE")
print("===================================")
print(f"Final dataset: {MERGED}")
print(f"Classes: {len(CLASS_NAMES)}")
print(f"YAML: {yaml_path}")