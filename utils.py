import torch 
import torchvision
import torchvision.utils
from dataset import CaravanaDataset
from torch.utils.data import DataLoader
import datetime

current_time = datetime.datetime.now().strftime("%d_%m_%Y")
current_timeL = datetime.datetime.now().strftime("%d_%m_%Y__%H:%M:%S")


def save_checkpoint(state, filename = f"checkpoint_{current_time}.pth.tar"):
    print("---------- Saving Checkpoint ------------")
    torch.save(state, filename)

    
def load_checkpoint(checkpoint, model):
    print("++++++++++ Loding Checkpoint +++++++++++++")
    model.load_state_dict(checkpoint["state_dict"])
     
## ------------------------------------- correction get loader----
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = CaravanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    val_ds = CaravanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,  
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory, 
        shuffle=False
    )

    return train_loader, val_loader

     
     
## Check Model accuracy       
def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels  = 0
    dice_score  = 0
    
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            x = x.permute(0, 3, 1, 2)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            acc = (num_correct/num_pixels) * 100
            
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-4)
            
            
            print(f"> {num_correct} out of {num_pixels}, with accuracy: {acc:.2f}")
            print(f"===> Dice Score: {dice_score/len(loader)}\n")
            
            model.train()


## Save predition images             
def save_predictionIMG(loader, model, folder = "outputs/",device = "cuda"):
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        
        x = x.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}_{current_timeL}.png")    
        
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    
    model.train()