from pathlib import Path
from collections import Counter
import yaml

# ============================================================
# PATH
# ============================================================

ROOT = Path(r"C:\GAURI\AI-Traffic-Violation-Detector")
DATASET = ROOT / "datasets" / "merged"

# ============================================================
# LOAD DATA.YAML
# ============================================================

yaml_path = DATASET / "data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

print("=" * 60)
print("MERGED DATASET VALIDATION")
print("=" * 60)

print(f"\nNumber of classes: {len(class_names)}")

print("\nClasses:")
for class_id, name in class_names.items():
    print(f"{class_id}: {name}")

# ============================================================
# VALIDATE EACH SPLIT
# ============================================================

overall_counts = Counter()

for split in ["train", "val", "test"]:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    print("\n" + "-" * 60)
    print(f"{split.upper()} DATASET")
    print("-" * 60)

    if not image_dir.exists():
        print(f"[ERROR] Missing image directory: {image_dir}")
        continue

    if not label_dir.exists():
        print(f"[ERROR] Missing label directory: {label_dir}")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    # --------------------------------------------------------
    # Check missing labels
    # --------------------------------------------------------

    missing_labels = []

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image.name)

    if missing_labels:

        print(
            f"\n[WARNING] Images without labels: "
            f"{len(missing_labels)}"
        )

        for name in missing_labels[:10]:
            print("   ", name)

    else:

        print("\n✓ Every image has a label")

    # --------------------------------------------------------
    # Check missing images
    # --------------------------------------------------------

    missing_images = []

    image_stems = {p.stem for p in images}

    for label in labels:

        if label.stem not in image_stems:
            missing_images.append(label.name)

    if missing_images:

        print(
            f"\n[WARNING] Labels without images: "
            f"{len(missing_images)}"
        )

        for name in missing_images[:10]:
            print("   ", name)

    else:

        print("✓ Every label has an image")

    # --------------------------------------------------------
    # Validate YOLO annotations
    # --------------------------------------------------------

    class_counts = Counter()

    invalid_lines = 0
    invalid_classes = 0
    invalid_coordinates = 0

    for label_path in labels:

        with open(label_path, "r") as f:

            for line_number, line in enumerate(f, start=1):

                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:

                    invalid_lines += 1

                    print(
                        f"[INVALID] {label_path.name} "
                        f"line {line_number}: "
                        f"expected 5 values"
                    )

                    continue

                try:

                    class_id = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    invalid_lines += 1

                    continue

                # Class validation

                if class_id < 0 or class_id >= len(class_names):

                    invalid_classes += 1

                    print(
                        f"[INVALID CLASS] "
                        f"{label_path.name}: "
                        f"{class_id}"
                    )

                    continue

                # Coordinate validation

                values = [x, y, w, h]

                if any(v < 0 or v > 1 for v in values):

                    invalid_coordinates += 1

                class_counts[class_id] += 1
                overall_counts[class_id] += 1

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id in range(len(class_names)):

        name = class_names[class_id]

        count = class_counts[class_id]

        print(
            f"{class_id:2d} | "
            f"{name:30s} | "
            f"{count}"
        )

    print("\nAnnotation problems:")

    print(f"Invalid lines: {invalid_lines}")
    print(f"Invalid class IDs: {invalid_classes}")
    print(f"Invalid coordinates: {invalid_coordinates}")


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("OVERALL CLASS DISTRIBUTION")
print("=" * 60)

for class_id in range(len(class_names)):

    print(
        f"{class_id:2d} | "
        f"{class_names[class_id]:30s} | "
        f"{overall_counts[class_id]}"
    )

print("\nValidation complete.")