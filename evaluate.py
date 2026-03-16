import os
import argparse
from ultralytics import YOLO
import wandb

def main():
    parser = argparse.ArgumentParser("Evaluate YOLOv10n on SARDet-100K")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to YOLO dataset config yaml (for class names and val/test split)")
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLOv10n_Eval", help="Wandb project name")
    parser.add_argument("--split", type=str, default="test", help="Which split to test on (val or test)")
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))

    model = YOLO(args.model)

    # Perform evaluation
    # To run on test set, make sure test images are configured in dataset.yaml
    metrics = model.val(data=args.data, split=args.split, plots=True)
    
    # Log specific metrics to wandb manually if needed
    wandb.log({
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    })

    wandb.finish()
    print(f"Evaluation completed. mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
