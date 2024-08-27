import torch 
import albumentations as ag 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.amp
from model_unet import UNET
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictionIMG,
)




## Specifying Hypermeters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1280 
IMAGE_WIDTH = 128   # 1918
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/processed/train_img_" 
TRAIN_MASK_DIR = "data/processed/train_mask_"
VAL_IMG_DIR = "data/processed/val_img_"
VAL_MASK_DIR = "data/processed/val_mask_"


# Train function
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)
        
        ## Forward
        with torch.amp.autocast('cuda'):
            prediction = model(data)
            loss = loss_fn(prediction, targets)
            
        ## Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss = loss.item())

# -----------------------------------------------------      
def main():
    train_transform = ag.Compose(
        [
                    ag.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
                    ag.Rotate(limit = 35, p = 1.0),
                    ag.HorizontalFlip(p = 0.5),
                    ag.VerticalFlip(p = 0.5),
                    ag.Normalize(
                        mean = [0.0, 0.0, 0.0],
                        std = [1.0, 1.0, 1.0],
                        max_pixel_value = 255.0,
                    ),
                    ToTensorV2(),
        ]
    )
    
    
    val_transforms = ag.Compose(
        [
            ag.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            ag.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0,
                )
            
        ]
    )
    
    model = UNET(in_channels = 3, out_channels = 1)
    model.to(device = DEVICE)
    
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) 
    
    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR, 
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR, 
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform, 
        val_transform=val_transforms,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )  
    
    # train_loader, val_loader = get_loaders(
    #                                             TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    #                                            VAL_IMG_DIR, VAL_MASK_DIR,
    #                                             train_transform, val_transforms,
    #                                             BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    #                                         )  # 
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_{current_time}.pth.tar"), model)
        
    
    check_accuracy(loader = val_loader, model = model, device = DEVICE)
    #scaler = torch.amp.grad_scaler('cuda')
    
    scaler_eval = torch.amp.GradScaler('cuda',init_scale=2**16, growth_factor=2, 
                                  backoff_factor=0.5, growth_interval=2000, 
                                  enabled=True)
    
    
    for epoch in range(NUM_EPOCHS):
        train_fn(loader = train_loader, model = model, 
                 optimizer = optimizer, loss_fn = loss, 
                 scaler = scaler_eval)    
        
        
        # model saving
        checkpoint = {
                        "state_dict" : model.state_dict(),
                        "optimizer"  : optimizer.state_dict()
                    }
        
        save_checkpoint(checkpoint)
        
        # accuracy 
        check_accuracy(loader = val_loader,model = model, device = DEVICE)
        
        # saving predictions 
        save_predictionIMG(val_loader, model, folder = "outputs/", device = DEVICE)
        
        
        
if __name__ == "__main__":
    main()
        