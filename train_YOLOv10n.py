import os
import argparse
from ultralytics import YOLO
import wandb

def main():
    parser = argparse.ArgumentParser("Train YOLOv10n on SARDet-100K")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to YOLO dataset config yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLOv10n", help="Wandb project name")
    parser.add_argument("--name", type=str, default="yolo", help="Run name")
    args = parser.parse_args()

    # Initialize W&B run
    wandb.init(project=args.project, name=args.name, config=vars(args))

    # Load a model
    model = YOLO(args.weights) # Initialize using specified weights

    # Train the model
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "device": "0", # For a single GPU via CUDA
        "patience": 50,
        "resume": args.resume,
        "lr0": args.lr0
    }
    
    if args.cfg:
        train_args["cfg"] = args.cfg

    results = model.train(**train_args)

    # Validate
    model.val()

    wandb.finish()

if __name__ == "__main__":
    main()
