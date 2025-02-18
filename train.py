import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import argparse
import torch
from graph_loss import GraphMatrixLoss
import sys

from DGL_datacore_endtoend import GraphDataset
from ViT_Graph_Transformer import GraphModel

def make_net(args):
    model_type = args.model
    if model_type == "Graph_ViT_Transformer":
        print("--Initialize model--")
        model = GraphModel()
    else:
        sys.exit("Model not found")

    if args.pretrained and os.path.exists(args.pretrained):
        try:
            checkpoint = torch.load(args.pretrained)
            state_dict =  {key.replace('module.', ''): value for key, value in checkpoint.items()}
            model.load_state_dict(state_dict, strict=False)
        except:
            pass

    if args.parallel:
        model = nn.DataParallel(model)

    device = torch.device(args.device)
    model.to(device)
    return model

def build_dataloaders(args):
    # For training in batches
    def collate_fn(batch):
        max_count = max(item[0].shape[0] for item, _ , _ in batch)
        batch_D_matrices = []
        batch_UD_matrices = []
        masks = []
        batch_images = []

        for UD_matrix, D_matrix, images in batch:

            current_count = D_matrix.shape[0]  

            if current_count < max_count:        
                missing_count = max_count - current_count       
                padding = torch.zeros(missing_count, 3, 224, 224, dtype=images.dtype, device=images.device)
                images_padded = torch.cat([images, padding], dim=0)
                padded_D_matrix = torch.nn.functional.pad(D_matrix, (0, missing_count, 0, missing_count))
                padded_UD_matrix = torch.nn.functional.pad(UD_matrix, (0, missing_count, 0, missing_count))

            else:
                images_padded = images
                padded_D_matrix = D_matrix
                padded_UD_matrix = UD_matrix

            batch_images.append(images_padded)
            batch_D_matrices.append(padded_D_matrix)
            batch_UD_matrices.append(padded_UD_matrix)

            mask = torch.ones(max_count, dtype=torch.bool)
            mask[current_count:] = False
            masks.append(mask)
        
        batch_D_matrices = torch.stack(batch_D_matrices)
        batch_UD_matrices = torch.stack(batch_UD_matrices)
        batch_images = torch.stack(batch_images)
        masks = torch.stack(masks)

        return batch_UD_matrices,batch_D_matrices, batch_images, masks
    
    print("--Initialize dataset--")

    dataset = GraphDataset(dataset=args.dataset, ratio = args.split_ratio)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])

    print("The len of training set: ", len(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    return train_dataloader,val_dataloader

def retrieve_original_data(padded_matrices, masks):
    original_matrices = []
    batch_size = padded_matrices.size(0)  

    for i in range(batch_size):
        current_mask = masks[i]  # This is a 1D mask
        valid_rows = padded_matrices[i][current_mask]
        valid_matrix = valid_rows[:, current_mask]   
        original_matrices.append(valid_matrix)

    return original_matrices

def retrieve_original_tensors(padded_tensors, masks):

    original_tensors = []
    batch_size = padded_tensors.size(0)
    
    for i in range(batch_size):
        current_mask = masks[i]
        valid_rows = padded_tensors[i][current_mask]
        
        original_tensors.append(valid_rows)
    
    return original_tensors

def train(model, train_dataloader, val_dataloader, criterion, optimizer, logging, args):
    train_dataloader = train_dataloader
    f_loss_fn = criterion
    num_epochs = args.epochs
    device = torch.device(args.device)

    for epoch in tqdm(range(num_epochs), desc='Training Progress', unit='iteration'):
        
        model.train()
        total_loss = 0
        for batch_UD_matrices, batch_D_matrices, batch_images, pad_masks in train_dataloader:
            optimizer.zero_grad()
            dist_matrices,direct_matrices = model(batch_images.to(device),batch_D_matrices.to(device),pad_masks)
            original_UD_path_matrices = retrieve_original_data(batch_UD_matrices.to(device), pad_masks)
            original_D_path_matrices = retrieve_original_data(batch_D_matrices.to(device), pad_masks)
            original_dist_matrices = retrieve_original_data(dist_matrices, pad_masks)
            original_direct_matrices = retrieve_original_data(direct_matrices, pad_masks)

            loss= f_loss_fn(original_UD_path_matrices, original_D_path_matrices,original_dist_matrices,original_direct_matrices)
            total_loss += loss

            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs},  Average Loss: {avg_loss:.4f}")
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs},  Average Loss: {avg_loss:.4f}")
        
        #validation
        if val_dataloader:
            model.eval()
            with torch.no_grad():
                for val_batch_UD_matrices, val_batch_D_matrices, val_batch_images, val_pad_masks in val_dataloader:
                    val_dist_matrices,val_direct_matrices = model(val_batch_images.to(device),None,val_pad_masks)
                    original_val_UD_path_matrices = retrieve_original_data(val_batch_UD_matrices.to(device),val_pad_masks)
                    original_val_D_path_matrices = retrieve_original_data(val_batch_D_matrices.to(device), val_pad_masks)
                    original_val_dist_matrices = retrieve_original_data(val_dist_matrices, val_pad_masks)
                    original_val_direct_matrices = retrieve_original_data(val_direct_matrices, val_pad_masks)
                    loss= f_loss_fn(original_val_UD_path_matrices, original_val_D_path_matrices,original_val_dist_matrices,original_val_direct_matrices)
                    total_loss += loss
            avg_loss = total_loss / len(val_dataloader)
            logging.info(
            f"Val-Average Loss: {avg_loss:.4f}")
            print(f"Val-Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.state_dict_path)
    print("--Model saved--")

def parse_args():
    datasets = ["NC2017_Dev_Ver1_Img","MFC18_Dev1_Image_Ver2", "Reddit"]
    parser = argparse.ArgumentParser(description='Training a model using timm and PyTorch.')
    parser.add_argument('--model', type=str, default='Graph_ViT_Transformer', help='Model architecture frsom timm library.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--dataset', type=str, default="NC2017_Dev_Ver1_Img", help='Path to the dataset.')
    parser.add_argument('--split_ratio', type=tuple, default=(0.05,0.95))
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--descript', type=str, default="")
    parser.add_argument('--cuda', type = str, default = "0,1,2,3")

    args, _ = parser.parse_known_args()
    parser.add_argument('--name', type=str, default= f"{args.dataset}_{args.model}_{args.descript}")

    args, _ = parser.parse_known_args()
    parser.add_argument('--state_dict_path', type=str, default= "/home/keyang/project/MyProvenance/Models/" + args.name + ".pth")
    parser.add_argument('--log_file_path', type=str, default= "/home/keyang/project/MyProvenance/Logs/"+ args.name +".log")
    parser.add_argument('--device', type=str, default= "cuda")
    parser.add_argument('--parallel', type=bool, default= True)

    args, _ = parser.parse_known_args()
    parser.add_argument('--pretrained', type=str, default=args.state_dict_path)
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    logging.basicConfig(filename=args.log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = make_net(args)  
    train_dataloader, val_dataloader = build_dataloaders(args)
    criterion = GraphMatrixLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                        lr = args.lr,
                        eps = 1e-08)
    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=criterion, optimizer=optimizer, logging=logging, args=args)

if __name__ == '__main__':
    main()
