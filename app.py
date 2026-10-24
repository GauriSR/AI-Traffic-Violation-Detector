
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Traffic Violation Detector",
    page_icon="🚦",
    layout="wide"
)


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():
    traffic_model = YOLO("models/best.pt")
    person_model = YOLO("yolo11n.pt")
    return traffic_model, person_model


traffic_model, person_model = load_models()


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box1[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(detections, iou_threshold=0.50):
    """
    Remove heavily overlapping detections, retaining the
    higher-confidence box.
    """

    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True
    )

    kept = []

    for detection in detections:
        duplicate = any(
            calculate_iou(detection["box"], previous["box"])
            >= iou_threshold
            for previous in kept
        )

        if not duplicate:
            kept.append(detection)

    return kept


def point_inside_box(x, y, box):
    """Check whether a point is inside a box."""

    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def phone_is_associated_with_person(phone_box, people):
    """Check whether a detected phone overlaps a detected person."""

    px1, py1, px2, py2 = phone_box
    center_x = (px1 + px2) / 2
    center_y = (py1 + py2) / 2

    for person in people:
        person_box = person["box"]

        if point_inside_box(center_x, center_y, person_box):
            return True

        if calculate_iou(phone_box, person_box) > 0:
            return True

    return False


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("🚦 AI Traffic Violation Detector")

st.write(
    "Upload a traffic image to detect potential helmet, "
    "mobile phone and triple-riding violations."
)

st.info(
    "Upload a traffic image and click Analyze Traffic. "
    "All results are model predictions and may require verification."
)


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a traffic image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")
    st.image(image, width=600)

    if st.button("🔍 Analyze Traffic", type="primary"):

        with st.spinner("Analyzing traffic image..."):

            # Custom model: traffic objects and violations
            traffic_result = traffic_model.predict(
                source=image,
                conf=0.05,
                iou=0.45,
                verbose=False
            )[0]

            # General YOLO model: people only
            person_result = person_model.predict(
                source=image,
                classes=[0],
                conf=0.35,
                iou=0.45,
                verbose=False
            )[0]

        # ==================================================
        # COLLECT DETECTIONS
        # ==================================================

        rider_candidates = []
        custom_phone_candidates = []
        bikes = []

        for box in traffic_result.boxes:

            class_id = int(box.cls[0])
            class_name = traffic_model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            coordinates = (x1, y1, x2, y2)

            if class_name in {"bike", "motorcycle"}:
                bikes.append(coordinates)

            elif class_name in {
                "driver_without_helmet",
                "passenger_without_helmet"
            }:
                rider_candidates.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": coordinates
                })

            elif class_name == "mobile_phone":
                custom_phone_candidates.append({
                    "box": coordinates,
                    "confidence": confidence,
                    "source": "Custom traffic model"
                })

        # ==================================================
        # DEDUPLICATE HELMET VIOLATIONS
        # ==================================================

        unique_rider_violations = remove_duplicate_detections(
            rider_candidates,
            iou_threshold=0.50
        )

        violation_counts = {
            "Driver without helmet": 0,
            "Passenger without helmet": 0,
            "Mobile phone usage": 0,
            "Triple riding": 0
        }

        violations = []

        for detection in unique_rider_violations:

            class_name = detection["class_name"]
            confidence = detection["confidence"]

            if class_name == "driver_without_helmet":
                violation_counts["Driver without helmet"] += 1

                violations.append(
                    f"🪖 Driver without helmet — {confidence:.0%}"
                )

            elif class_name == "passenger_without_helmet":
                violation_counts["Passenger without helmet"] += 1

                violations.append(
                    f"🪖 Passenger without helmet — {confidence:.0%}"
                )

        # ==================================================
        # PERSON DETECTION
        # ==================================================

        people = []

        for box in person_result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            people.append({
                "box": (x1, y1, x2, y2),
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "confidence": float(box.conf[0])
            })

        # ==================================================
        # PHONE DETECTION — CUSTOM MODEL ONLY
        # ==================================================

        unique_phone_candidates = remove_duplicate_detections(
            custom_phone_candidates,
            iou_threshold=0.30
        )

        associated_phones = [
            phone
            for phone in unique_phone_candidates
            if phone_is_associated_with_person(
                phone["box"],
                people
            )
        ]

        violation_counts["Mobile phone usage"] = len(associated_phones)

        for phone in associated_phones:
            violations.append(
                "📱 Potential phone-use violation — "
                f"{phone['confidence']:.0%} confidence"
            )

        # ==================================================
        # TRIPLE-RIDING DETECTION
        # ==================================================

        triple_riding_detected = False
        max_people_on_bike = 0

        for bike in bikes:

            bx1, by1, bx2, by2 = bike

            bike_width = bx2 - bx1
            bike_height = by2 - by1

            # Expanded region around the motorcycle
            rx1 = bx1 - bike_width * 0.35
            rx2 = bx2 + bike_width * 0.35
            ry1 = by1 - bike_height * 2.0
            ry2 = by2 + bike_height * 0.25

            people_on_bike = 0

            for person in people:

                px = person["center_x"]
                py = person["center_y"]

                if (
                    rx1 <= px <= rx2
                    and ry1 <= py <= ry2
                ):
                    people_on_bike += 1

            max_people_on_bike = max(
                max_people_on_bike,
                people_on_bike
            )

            if people_on_bike >= 3:
                triple_riding_detected = True

        if triple_riding_detected:

            violation_counts["Triple riding"] = 1

            violations.append(
                "👥 Triple riding detected — "
                f"{max_people_on_bike} people associated with a bike"
            )

        # ==================================================
        # DISPLAY RESULTS
        # ==================================================

        st.subheader("🔍 Traffic Detection Result")
        st.image(traffic_result.plot(), width=600)

        st.subheader("👥 Person Detection")
        st.metric("People detected in image", len(people))

        st.subheader("🚨 Traffic Violation Analysis")

        total_violations = sum(violation_counts.values())

        if total_violations > 0:

            st.metric("Total Potential Violations", total_violations)

            st.write("### Violation Summary")

            for name, count in violation_counts.items():
                if count > 0:
                    st.write(f"🔴 **{name}: {count}**")

            st.write("### 📋 Detection Details")

            for violation in violations:
                st.warning(violation)

        else:
            st.success("No potential violations detected.")

        # ==================================================
        # DEBUG DETAILS
        # ==================================================

        with st.expander("🔎 Debug Detection Details"):

            st.write("**Person detections:**")

            for index, person in enumerate(people, 1):

                rounded_box = tuple(
                    round(value, 1)
                    for value in person["box"]
                )

                st.write(
                    f"Person {index}: "
                    f"confidence={person['confidence']:.0%}, "
                    f"box={rounded_box}"
                )

            st.write("**Detected bikes/motorcycles:**")
            st.write(len(bikes))

            st.write("**Raw helmet detections:**")
            st.write(len(rider_candidates))

            st.write("**Helmet detections after deduplication:**")
            st.write(len(unique_rider_violations))

            st.write("**Custom-model phone detections:**")
            st.write(len(custom_phone_candidates))

            st.write("**Unique phone candidates:**")
            st.write(len(unique_phone_candidates))

            st.write("**Phone candidates associated with people:**")
            st.write(len(associated_phones))

            st.write("**Maximum people associated with one bike:**")
            st.write(max_people_on_bike)

            st.caption(
                "Phone association is based on bounding-box overlap. "
                "A nearby phone is not proof of phone use while riding. "
                "Triple-riding counts are approximate and may include "
                "nearby people."
            )
