
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("../models/best.pt")

# Test image
image_path = "../test.jpg"

# Run detection
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Violation rules
violations = []

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        # Helmet violation
        if class_name == "driver_without_helmet":
            violations.append(
                f"Helmet Violation - Driver (confidence: {confidence:.2f})"
            )

        elif class_name == "passenger_without_helmet":
            violations.append(
                f"Helmet Violation - Passenger (confidence: {confidence:.2f})"
            )

        # Mobile phone violation
        elif class_name == "mobile_phone":
            violations.append(
                f"Mobile Phone Violation (confidence: {confidence:.2f})"
            )


# Display results
print("\n" + "=" * 50)
print("TRAFFIC VIOLATION ANALYSIS")
print("=" * 50)

if violations:
    print("\nViolations detected:")

    for violation in violations:
        print(f"🚨 {violation}")

else:
    print("\n✅ No traffic violations detected.")

print("=" * 50)

