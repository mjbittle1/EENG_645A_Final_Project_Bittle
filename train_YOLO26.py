import os
import argparse
import wandb
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO26 model using DDP on the cluster.")
    parser.add_argument("--size", type=str, choices=['n', 's', 'm', 'l', 'x'], required=True, 
                        help="YOLO26 model size variant to train ('n', 's', 'm', 'l', or 'x')")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to YOLO dataset config yaml")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. Default is -1 (AutoBatch) to maximize GPU memory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLO26", help="Wandb project name")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="GPUs to use for DDP (e.g., '0' or '0,1,2,3')")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--cache", action="store_true", help="Cache images into RAM to accelerate training")
    args = parser.parse_args()

    model_name = f"yolo26{args.size}"
    weights = f"{model_name}.pt"

    print(f"\n{'='*50}")
    print(f"Initializing training for {model_name.upper()}")
    print(f"Reserving cluster resources: DDP across device(s) '{args.device}' with batch size {args.batch}.")
    print(f"{'='*50}\n")

    # Manually init wandb so we can log args properly
    wandb.init(project=args.project, name=model_name, config=vars(args))

    # Initialize YOLO model
    model = YOLO(weights)
    
    # Train
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": model_name,
        "device": args.device, # Will trigger DDP if multiple GPUs are specified
        "patience": args.patience,
        "cache": args.cache,
    }
    
    model.train(**train_args)
    
    # Run validation explicitly (optional, it also validates during training)
    model.val()
    
    wandb.finish()

if __name__ == "__main__":
    main()
