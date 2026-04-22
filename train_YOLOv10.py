import os
import argparse
import wandb
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv10 model using DDP on the cluster.")
    
    # UPDATED: Swapped to YOLOv10 and added the 'b' (balanced) variant specific to v10 architectures
    parser.add_argument("--size", type=str, choices=['n', 's', 'm', 'b', 'l', 'x'], required=True, 
                        help="YOLOv10 model size variant to train ('n', 's', 'm', 'b', 'l', or 'x')")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to YOLO dataset config yaml")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    
    # Increased default batch size to 512 (64 per GPU) for the H200s
    parser.add_argument("--batch", type=int, default=512, help="Total batch size across all GPUs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--cfg", type=str, default=None, help="Path to tuned hyperparameters yaml file")
    
    # Default to 'auto' to ensure MuSGD is used for rapid early convergence
    parser.add_argument("--optimizer", type=str, default="auto", choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="Optimizer to use")
    
    # KEPT SAME: Keeping project default identical so YOLOv10 runs pipe into the existing dashboard
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLO26", help="Wandb project name")
    
    # Default expanded to use all 8 GPUs
    parser.add_argument("--device", type=str, default="0,1,2,3,4,5,6,7", help="GPUs to use for DDP (e.g., '0' or '0,1,2,3,4,5,6,7')")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--cache", action="store_true", help="Cache images into RAM to accelerate training")
    args = parser.parse_args()

    # UPDATED: Targets the yolov10 weight namespace
    model_name = f"yolov10{args.size}"
    weights = f"{model_name}.pt"

    print(f"\n{'='*50}")
    print(f"Initializing training for {model_name.upper()}")
    print(f"Reserving cluster resources: DDP across device(s) '{args.device}' with batch size {args.batch}.")
    print(f"{'='*50}\n")

    # Setup DDP Node Awareness
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Force W&B to use threads instead of multiprocessing to avoid PyTorch DDP conflicts
    os.environ["WANDB_START_METHOD"] = "thread"

    # Initialize W&B strictly on the Master GPU node to prevent DDP socket clashes
    if local_rank in [-1, 0]:
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
        "device": args.device, # Pass the device array natively so YOLO doesn't crash torch.cuda.set_device
        "patience": args.patience,
        "cache": args.cache,
        "workers": 0, 
        "optimizer": args.optimizer,
        "degrees": 180.0, # Satellite augmentation
        "flipud": 0.5,    # Satellite augmentation
    }
    
    if args.cfg:
        train_args["cfg"] = args.cfg
        
    model.train(**train_args)
    
    # Run validation explicitly (optional, it also validates during training)
    model.val()
    
    if local_rank in [-1, 0]:
        wandb.finish()

if __name__ == "__main__":
    main()