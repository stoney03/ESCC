import os
os.environ["OMP_NUM_THREADS"]="10"
import torch
import torch.nn.functional as F
import timm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import argparse
from datasets.dataset_patch_feat import PatchDataset
import math
import torch.cuda.amp as amp
import time
from scipy.stats import pearsonr



class MultiHeadAttentionMILAggregator(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads=8, conv_kernel_size=3, conv_out_channels=512):

        super(MultiHeadAttentionMILAggregator, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.conv = nn.Conv1d(in_channels=num_patches, out_channels=conv_out_channels, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.unsqueeze(0)

        attn_output, attn_weights = self.attn(x, x, x)

        attention_pool = torch.bmm(attn_weights, attn_output)
        attention_pool = attention_pool.squeeze(0)

        conv_out = self.conv(attention_pool)
        aggregated_feature = conv_out.mean(dim=0).unsqueeze(0)

        return aggregated_feature
    


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim, input_size, patch_size):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.row_col_embeddings = self.create_position_encoding(input_size, patch_size, embed_dim).to(self.position_embeddings.device)

    def create_position_encoding(self, input_size, patch_size, embed_dim):
        num_patches_h = input_size // patch_size
        num_patches_w = input_size // patch_size
        num_patches = num_patches_h * num_patches_w

        row_idx = torch.arange(0, num_patches_h).repeat(num_patches_w, 1).t().flatten()
        col_idx = torch.arange(0, num_patches_w).repeat(num_patches_h, 1).flatten()

        position_embeddings = torch.zeros((num_patches, embed_dim))

        position_embeddings[:, :embed_dim // 2] = row_idx.unsqueeze(1) / (num_patches_h - 1)
        position_embeddings[:, embed_dim // 2:] = col_idx.unsqueeze(1) / (num_patches_w - 1)

        return position_embeddings.unsqueeze(0)

    def forward(self, x):
        self.row_col_embeddings = self.row_col_embeddings.to(self.position_embeddings.device)

        pos_embeddings = self.position_embeddings + self.row_col_embeddings

        return x + pos_embeddings.squeeze(0)



def load_model(args):
    
    if args.model == 'vit_large_uni':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load("/pytorch_model/UNI_pytorch_model.bin"))

    elif args.model == 'vit_small_patch8_224_dino':
        model = timm.create_model('vit_small_patch8_224.dino', pretrained=False)
        model.load_state_dict(torch.load("/pytorch_model/vit_small_patch8_224_dino.bin"))
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_features = module.in_features
                break
        model.head = nn.Linear(num_features, num_classes=0)

    elif args.model == 'prov_gigapath':
        pretrained_cfg = {"tag": "", "custom_load": False, "input_size": [3, 224, 224], "fixed_input_size": True, "interpolation": "bicubic", "crop_pct": 1.0, "crop_mode": "center", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 0, "pool_size": None, "first_conv": "patch_embed.proj", "classifier": "head", "license": "prov-gigapath"}
        model_args = {"img_size": 224, "in_chans": 3, "patch_size": 16, "embed_dim": 1536, "depth": 40, "num_heads": 24, "init_values": 1e-05, "mlp_ratio": 5.33334, "num_classes": 0}
        model = timm.create_model(model_name="vit_giant_patch14_dinov2", 
                                 pretrained=False,
                                 pretrained_cfg=pretrained_cfg,
                                 global_pool="token",
                                 **model_args)
        model.load_state_dict(torch.load("/pytorch_model/Prov-GigaPath_tile.bin"))

    elif args.model == 'virchow':
        pretrained_cfg = {
            "tag": "virchow_v1", "custom_load": False, "input_size": [3, 224, 224], "fixed_input_size": False, "interpolation": "bicubic", "crop_pct": 1.0, "crop_mode": "center", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 0, "pool_size": None, "first_conv": "patch_embed.proj", "classifier": "head", "license": "Apache 2.0"}
        model_args= {
            "img_size": 224, "init_values": 1e-5, "num_classes": 0, "mlp_ratio": 5.3375, "global_pool": "", "dynamic_img_size": True}
        model = timm.create_model(model_name="vit_huge_patch14_224",
                          pretrained=False,
                          pretrained_cfg=pretrained_cfg,
                          mlp_layer=timm.layers.SwiGLUPacked, 
                          act_layer=torch.nn.SiLU,
                          **model_args)
        model.load_state_dict(torch.load("/pytorch_model/Virchow.bin"))

    return model


def freeze_model(model, freeze_layers):
    layer_idx = 0
    for child in model.children():
        if layer_idx < freeze_layers:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        layer_idx += 1
    return model


def compute_clinical_similarity(clinical_features):
    norms = np.linalg.norm(clinical_features, axis=1, keepdims=True)
    similarity_matrix = np.dot(clinical_features, clinical_features.T)
    similarity_matrix /= (norms * norms.T)
    return similarity_matrix


def split_into_patches(image, patch_size):

    channels, img_height, img_width = image.shape

    patches = F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
    patches = patches.permute(0, 2, 1)
    patches = patches.view(-1, channels, patch_size, patch_size)

    return patches


def contrastive_loss(image_features, similarity_matrix, clinical_similarity_matrix, margin=1.0):

    num_pairs = image_features.size(0)
    flag = True
    if image_features.size(0) == 1:
        flag = False
        return flag, 1
    loss = 0

    for i in range(num_pairs):
        for j in range(i + 1, num_pairs):
            image_similarity = similarity_matrix[i, j]

            clinical_similarity = clinical_similarity_matrix[i, j]

            loss = (clinical_similarity-image_similarity) ** 2

    return flag, loss / (num_pairs * (num_pairs - 1) / 2)


def compute_similarity(feature_1, feature_2):
    feature_1 = feature_1.squeeze()
    feature_2 = feature_2.squeeze()

    similarity = F.cosine_similarity(feature_1, feature_2, dim=0)
    return similarity


def compute_similarity_matrix(features):

    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.mm(features_norm, features_norm.t())
    return sim_matrix


def extract_image_features(model, images):
    return model(images)


def train(model, train_loader, optimizer, num_epochs, freeze_layers, patch_size, input_size):
    model = freeze_model(model, freeze_layers)
    model.train()
    
    num_patches = (input_size // patch_size) ** 2
    if args.model == 'virchow': embed_dim = 1280
    elif args.model == 'prov_gigapath': embed_dim = 1536
    else: embed_dim = 1024

    position_encoder = TrainablePositionalEncoding(num_patches, embed_dim, input_size, patch_size).to(device)
    mil_aggregator = MultiHeadAttentionMILAggregator(num_patches, embed_dim).to(device)

    all_image_similarities = []
    all_clinical_similarities = []

    for epoch in range(num_epochs):
        total_loss = 0

        if epoch == num_epochs - 1:
            all_image_similarities = []
            all_clinical_similarities = []

        epoch_patch_time = 0.0
        epoch_position_time = 0.0
        epoch_aggregation_time = 0.0
        epoch_start_time = time.time()

        for images, clinical_features in train_loader:

            images = images.to(device)
            clinical_features = clinical_features.to(device)

            batch_aggregated_features = []

            clinical_features_norm = clinical_features / (clinical_features.norm(dim=1, keepdim=True) + 1e-8)
            clinical_similarity_matrix = torch.mm(clinical_features_norm, clinical_features_norm.t())

            for image in images:

                start_time_patch = torch.cuda.Event(enable_timing=True)
                end_time_patch = torch.cuda.Event(enable_timing=True)
                
                start_time_patch.record()
                patches = split_into_patches(image, patch_size)
                patches = patches.to(device)
                end_time_patch.record()
                torch.cuda.synchronize()
                patch_time = start_time_patch.elapsed_time(end_time_patch) / 1000.0
                epoch_patch_time += patch_time

                patch_features = extract_image_features(model, patches)


                start_time_position = torch.cuda.Event(enable_timing=True)
                end_time_position = torch.cuda.Event(enable_timing=True)
                start_time_position.record()
                patch_features = position_encoder(patch_features)
                end_time_position.record()
                torch.cuda.synchronize()
                position_time = start_time_position.elapsed_time(end_time_position) / 1000.0
                epoch_position_time += position_time


                start_time_aggregation = torch.cuda.Event(enable_timing=True)
                end_time_aggregation = torch.cuda.Event(enable_timing=True)
                start_time_aggregation.record()
                aggregated_feature = mil_aggregator(patch_features)
                end_time_aggregation.record()
                torch.cuda.synchronize()
                aggregation_time = start_time_aggregation.elapsed_time(end_time_aggregation) / 1000.0
                epoch_aggregation_time += aggregation_time

                batch_aggregated_features.append(aggregated_feature)

            batch_aggregated_features = torch.cat(batch_aggregated_features, dim=0)
            
            similarity_matrix = compute_similarity_matrix(batch_aggregated_features)

            if epoch == num_epochs - 1:
                all_image_similarities.append(similarity_matrix.detach().cpu().numpy())
                all_clinical_similarities.append(clinical_similarity_matrix.detach().cpu().numpy())

            optimizer.zero_grad()

            flag, loss = contrastive_loss(batch_aggregated_features, similarity_matrix, clinical_similarity_matrix, margin=1.0)
            if flag:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


        epoch_total_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")
        print(f"Total time in this epoch:")
        print(f"  - Total epoch time: {epoch_total_time:.2f}s")
        print(f"  - * Patch subdivision: {epoch_patch_time:.4f}s")
        print(f"  - * Position encoding: {epoch_position_time:.4f}s")
        print(f"  - * Aggregation: {epoch_aggregation_time:.4f}s")
        print(f"  - Total(*): {epoch_patch_time + epoch_position_time + epoch_aggregation_time:.4f}s")


        os.makedirs(args.output, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'position_encoder_state_dict': position_encoder.state_dict(),
            'mil_aggregator_state_dict': mil_aggregator.state_dict(),
        }, f"{args.output}/model_epoch_{epoch + 1}.pth")


    if num_epochs > 0 and len(all_image_similarities) > 0:

        image_similarities = np.concatenate(all_image_similarities, axis=0)
        clinical_similarities = np.concatenate(all_clinical_similarities, axis=0)
        
        image_sim_flat = image_similarities.flatten()
        clinical_sim_flat = clinical_similarities.flatten()
        
        corr_coef, p_value = pearsonr(image_sim_flat, clinical_sim_flat)
        


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--gpus', type=int, default=0, help="GPU indices, comma separated, e.g. '0,1' ")
    parser.add_argument('--output', type=str, default="", help="Output directory to save models and logs.")
    parser.add_argument('--logger_name', default=None, help="Name of logging.getLogger(name) for record.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train.")
    parser.add_argument('--csv_file', type=str, default='./patch_feat_paths.csv', help="CSV file path with patch and TME feature.")
    parser.add_argument('--num_samples', type=int, default=15000, help="Number of samples to train on.")
    parser.add_argument('--freeze_layers', type=int, default=300, help="Number of layers to freeze during training.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--input_size', type=int, default=1120, help="Size of each patch.")
    parser.add_argument('--patch_size', type=int, default=224, help="Size of each sub-patch.")
    parser.add_argument('--model', default='vit_large_uni', help="Fine-tune model name.")
    return parser.parse_args()


if __name__ == "__main__":

    args = args_parser()
    device = torch.device(args.gpus)

    args.output = f'/result/{args.model}/input_size{args.input_size}/freeze{args.freeze_layers}/samples{args.num_samples}/'
    
    model = load_model(args)
    model = model.to(device)

    if args.model in ['vit_large_uni','vit_small_patch8_224_dino']:
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif args.model == 'prov_gigapath':
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif args.model == 'virchow':
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


    train_data = PatchDataset(args.csv_file, transform=transform)
    train_sampler = SubsetRandomSampler(np.random.choice(len(train_data), args.num_samples))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train(model, train_loader, optimizer, args.epochs, args.freeze_layers, args.patch_size, args.input_size)
