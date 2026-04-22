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
    
    # UPDATED: Reduced default batch size from 512 to 128. 
    # This yields 16 images per GPU on an 8-GPU cluster, massively increasing your gradient steps per epoch.
    parser.add_argument("--batch", type=int, default=128, help="Total batch size across all GPUs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--cfg", type=str, default=None, help="Path to tuned hyperparameters yaml file")
    
    parser.add_argument("--optimizer", type=str, default="auto", choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="Optimizer to use")
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLO26", help="Wandb project name")
    parser.add_argument("--name", type=str, default="", help="Run name (optional)")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM")
    parser.add_argument("--device", type=str, default="0,1,2,3,4,5,6,7", help="CUDA device ids (e.g., '0,1,2,3' or '0')")
    
    args = parser.parse_args()

    model_name = args.name if args.name else f"YOLO26{args.size.upper()}"
    weights = f"yolo26{args.size}.pt" 

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
        "device": args.device,
        "patience": args.patience,
        "cache": args.cache,
        "optimizer": args.optimizer,
        "workers": 0        
        # UPDATED: Removed "workers": 0. 
        # By removing this, Ultralytics will automatically assign background worker threads 
        # to load data, preventing the CPU bottleneck that starved your GPUs.
        
        # UPDATED: Removed "degrees": 180.0 and "flipud": 0.5.
        # This reverts the model to standard baseline augmentations for early stability.
    }
    
    if args.cfg:
        train_args["cfg"] = args.cfg
        
    model.train(**train_args)

if __name__ == "__main__":
    main()