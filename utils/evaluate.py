import json
import os

from shapely import wkt
from sklearn.metrics import classification_report, confusion_matrix

GT_DIR = "/Users/paolo/Desktop/train/labels"
PRED_DIR = "/Users/paolo/Desktop/output_batch"
IOU_THRESHOLD = 0.5


def load_features(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["features"]["xy"]


def prediction_base_to_gt_base(file_name: str) -> str:
    base = file_name[:-5]

    if base.endswith("_loc"):
        base = base[:-4]
    elif base.endswith("_cls"):
        base = base[:-4]

    return base


def polygon_iou(poly_a, poly_b):
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return inter / union


def build_polygon_entries(features, is_ground_truth: bool):
    entries = []

    for item in features:
        props = item.get("properties", {})
        subtype = props.get("subtype")
        poly_wkt = item.get("wkt")

        if subtype is None or poly_wkt is None:
            continue

        try:
            geom = wkt.loads(poly_wkt)
        except Exception:
            continue

        if geom.is_empty:
            continue

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            continue

        entries.append(
            {
                "subtype": subtype,
                "geometry": geom,
            }
        )

    return entries


def match_predictions_to_ground_truth(gt_entries, pred_entries, iou_threshold=0.5):
    y_true = []
    y_pred = []
    used_gt = set()

    for pred_idx, pred in enumerate(pred_entries):
        best_gt_idx = None
        best_iou = 0.0

        for gt_idx, gt in enumerate(gt_entries):
            if gt_idx in used_gt:
                continue

            iou = polygon_iou(pred["geometry"], gt["geometry"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            y_true.append(gt_entries[best_gt_idx]["subtype"])
            y_pred.append(pred["subtype"])

    return y_true, y_pred, len(used_gt)


def main():
    all_y_true = []
    all_y_pred = []
    processed_files = 0
    skipped_files = 0

    if not os.path.isdir(GT_DIR):
        raise FileNotFoundError(f"Ground truth folder not found: {GT_DIR}")

    if not os.path.isdir(PRED_DIR):
        raise FileNotFoundError(f"Prediction folder not found: {PRED_DIR}")

    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".json")])

    if not pred_files:
        print(f"No prediction JSON files found in: {PRED_DIR}")
        return

    for file_name in pred_files:
        base = prediction_base_to_gt_base(file_name)
        gt_file = os.path.join(GT_DIR, base + "_post_disaster.json")
        pred_file = os.path.join(PRED_DIR, file_name)

        if not os.path.exists(gt_file):
            print(f"Skipping {file_name}: missing ground truth file {gt_file}")
            skipped_files += 1
            continue

        try:
            gt_features = load_features(gt_file)
            pred_features = load_features(pred_file)
        except Exception as e:
            print(f"Skipping {file_name}: failed to read JSON ({e})")
            skipped_files += 1
            continue

        gt_entries = build_polygon_entries(gt_features, is_ground_truth=True)
        pred_entries = build_polygon_entries(pred_features, is_ground_truth=False)

        y_true, y_pred, matched = match_predictions_to_ground_truth(
            gt_entries, pred_entries, iou_threshold=IOU_THRESHOLD
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(
            f"{file_name}: gt_buildings={len(gt_entries)}, pred_buildings={len(pred_entries)}, matched={matched}"
        )
        processed_files += 1

    if not all_y_true:
        print("No matched building predictions found. Nothing to evaluate.")
        return

    labels = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    correct = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p)
    accuracy = correct / len(all_y_true)

    print("\n=== Summary ===")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Matched buildings: {len(all_y_true)}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            all_y_true,
            all_y_pred,
            labels=labels,
            zero_division=0,
            digits=4,
        )
    )

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    print("Labels order:", labels)
    print(cm)


if __name__ == "__main__":
    main()