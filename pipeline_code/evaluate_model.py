from ultralytics import YOLO

model = YOLO("trained_model/best.pt")

# Run validation
results = model.val(data="datasets/data.yaml")

print("\n===== VALIDATION METRICS =====")

# These are the correct properties
print("Classes:", results.names)
print("Precision per class:", results.box.p)
print("Recall per class:", results.box.r)
print("F1-score per class:", results.box.f1)
print("AP50 per class:", results.box.ap50)
print("mAP50-95 per class:", results.box.ap)

# Summary metrics
print("\n--- SUMMARY ---")
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
print("Mean Precision:", results.box.mp)
print("Mean Recall:", results.box.mr)

# Write metrics to file
with open("training_logs/validation_metrics.txt", "w") as f:
    f.write("===== VALIDATION METRICS =====\n")
    f.write(f"Classes: {results.names}\n")
    f.write(f"Precision per class: {results.box.p}\n")
    f.write(f"Recall per class: {results.box.r}\n")
    f.write(f"F1-score per class: {results.box.f1}\n")
    f.write(f"AP50 per class: {results.box.ap50}\n")
    f.write(f"mAP50-95 per class: {results.box.ap}\n\n")
    f.write("--- SUMMARY ---\n")
    f.write(f"mAP50: {results.box.map50}\n")
    f.write(f"mAP50-95: {results.box.map}\n")
    f.write(f"Mean Precision: {results.box.mp}\n")
    f.write(f"Mean Recall: {results.box.mr}\n")
