import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from sklearn.manifold import TSNE

from data.datasets import ImageFolder
from models import resnet
from models.SlotFA import SlotFAEval

from torch.utils.data import DataLoader
from tqdm import tqdm

def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = (img * std[:, None, None] + mean[:, None, None]) * 255.0
    return img.permute(1, 2, 0).cpu().type(torch.uint8)

def get_model(args):
    """ 載入並建立 SlotConEval 模型 """
    encoder = resnet.__dict__[args.arch]
    model = SlotFAEval(encoder, args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # 移除 "module." 這類前綴
    weights = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model

def get_features_and_assign(model, dataset, bs):
    """
    跑一次整個資料集，取出:
      1. 特徵 (flatten)
      2. 該特徵所對應的 slot assignment
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False
    )

    all_feats = []
    all_slots = []
    all_img_idx = []
    
    for data in tqdm(loader, desc='Extracting features & slots', leave=False):
        data = data.cuda(non_blocking=True)
        B = data.shape[0]
        
        # 1) 取得特徵: (B, d, H, W)
        with torch.no_grad():
            feat_2d = model.projector_k(model.encoder_k(data))  # (B, dim_out, H, W)
            feat_2d = F.normalize(feat_2d, dim=1)

            # 2) 取得 slot assignment
            #    grouping_k(feat_2d) -> (slots, probs)
            #    probs shape: (B, num_prototypes, H, W)
            _, probs = model.grouping_k(feat_2d)
            slot_assign = probs.argmax(dim=1)  # (B, H, W)

        # Flatten 特徵, shape: (B*H*W, dim_out)
        B, d, H, W = feat_2d.shape
        feats_flat = feat_2d.permute(0, 2, 3, 1).reshape(-1, d)  

        # Flatten slot assignment, shape: (B*H*W,)
        slots_flat = slot_assign.view(-1)
        image_idx_flat = torch.arange(B, device=data.device).repeat_interleave(H*W)

        all_feats.append(feats_flat.cpu())
        all_slots.append(slots_flat.cpu())
        all_img_idx.append(image_idx_flat.cpu())

    all_feats = torch.cat(all_feats, dim=0)  # (N, d)
    all_slots = torch.cat(all_slots, dim=0)  # (N,)
    all_img_idx = torch.cat(all_img_idx, dim=0)  # (N,)
    return all_feats, all_slots, all_img_idx

def visualize_tsne(all_feats, all_slots, all_img_idx, args):
    """
    對所有 flatten 特徵做 t-SNE，並根據 slot assignment 上色
    """

    matplotlib.rcParams.update({
        'font.size': 14,       # 文字基本大小
        'axes.labelsize': 16,  # x, y 軸標籤大小
        'axes.titlesize': 16,  # 標題大小
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    # 1. 轉成 numpy
    features_np = all_feats.numpy()
    slots_np = all_slots.numpy()
    img_idx_np = all_img_idx.numpy()

    # 2. t-SNE
    print("Running t-SNE on total {} points...".format(features_np.shape[0]))
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, init='random')
    feats_2d = tsne.fit_transform(features_np)  # (N, 2)

    # 3. 繪圖
    plt.figure(figsize=(4, 3))
    sc = plt.scatter(
        feats_2d[:, 0],
        feats_2d[:, 1],
        c=slots_np,            # 用 slot ID 上色
        cmap='tab20',          # 或 'rainbow', 'tab10', ...
        alpha=0.5,
        s=30                    # 每個點的大小
    )

    plt.axis('off')
    unique_imgs = np.unique(img_idx_np)
    for img_i in unique_imgs:
        subset = (img_idx_np == img_i)
        if np.sum(subset) == 0:
            continue
        x_mean = feats_2d[subset, 0].mean()
        y_mean = feats_2d[subset, 1].mean()
        offset_x = 0  # 你可依情況調大/調小
        plt.text(x_mean + offset_x, y_mean,  # 只加 x 偏移
                 f"{img_i}", 
                 fontsize=20, color="k")

    # 4. 儲存或顯示
    # plt.savefig(args.save_path, bbox_inches='tight', dpi=args.dpi)
    # print(f"Saved t-SNE figure to {args.save_path}")
    # 若想要互動式顯示，可加上 plt.show()
    plt.tight_layout()
    plt.show()

def count_distinct_slots_per_image(model, dataset, batch_size):
    """
    回傳一個 shape=(len(dataset),) 的 numpy array，
    其中 distinct_count_all[i] = 第 i 張影像用到幾個 slot。
    """

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    distinct_count_all = np.zeros(len(dataset), dtype=np.int32)  # 預先空間

    offset = 0  # 用來記錄目前處理到第幾張 image (全域)

    for data in tqdm(loader, desc='Counting distinct slots per image'):
        B = data.size(0)
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            feat_2d = model.projector_k(model.encoder_k(data))  # (B, dim_out, H, W)
            feat_2d = F.normalize(feat_2d, dim=1)
            _, probs = model.grouping_k(feat_2d)  # (B, num_slots, H, W)
            slot_assign = probs.argmax(dim=1)     # (B, H, W)

        # slot_assign[i] -> shape (H, W), 第 i 張影像
        # 找出 distinct slots
        slot_assign_np = slot_assign.view(B, -1).cpu().numpy()  # (B, H*W)

        for i in range(B):
            used_slots = np.unique(slot_assign_np[i])  # array of distinct slot IDs
            distinct_count_all[offset + i] = len(used_slots)

        offset += B

    return distinct_count_all

def count_slots_and_save(model, dataset, batch_size, output_path):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    slot_counts = []

    for data in tqdm(loader, desc='Counting slots for each image'):
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            feat_2d = model.projector_k(model.encoder_k(data))  # (B, dim_out, H, W)
            feat_2d = torch.nn.functional.normalize(feat_2d, dim=1)
            _, probs = model.grouping_k(feat_2d)  # (B, num_slots, H, W)
            slot_assign = probs.argmax(dim=1)  # (B, H, W)

        for i in range(data.size(0)):
            used_slots = torch.unique(slot_assign[i]).cpu().numpy()  # array of distinct slot IDs
            slot_counts.append(len(used_slots))

    # Save to txt file
    with open(output_path, 'w') as f:
        for count in slot_counts:
            f.write(f"{count}\n")

    print(f"Saved slot counts to {output_path}")

def plot_distinct_slots(distinct_count_all):
    import numpy as np
    import matplotlib.pyplot as plt

    counts = np.bincount(distinct_count_all)
    max_slots = len(counts) - 1
    
    x = np.arange(max_slots + 1)  # x 軸從 0 ~ max_slots
    y = counts

    avg_slots = np.mean(distinct_count_all)
    median_slots = np.median(distinct_count_all)
    print(f"Average distinct slots used per image: {avg_slots:.2f}")
    print(f"Median distinct slots used per image: {median_slots:.2f}")
    plt.figure(figsize=(6, 4))

    # 關鍵：width=1.0, align='edge' 讓長條彼此緊貼
    plt.bar(
        x,
        y,
        color='skyblue',
        edgecolor='black',
        width=1.0,       # 讓條寬就是 1
        align='edge'     # 從 x 的左側開始畫
    )

    # 如果想要在軸上對應每個整數，都顯示刻度，可以：
    # plt.xticks(x)  # 或只顯示每 2、5 個刻度: plt.xticks(x[::2]) / x[::5]
    plt.xticks(range(0, max_slots + 1, 2))

    # 隱藏上、右邊框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('Number of distinct slots used by image')
    plt.ylabel('Number of images')
    plt.title(
        f"COCO val Slot Activate Distribution)"
    )
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='COCOval')
    parser.add_argument('--data_dir', type=str, default='./datasets/coco')
    parser.add_argument('--batch_size', type=int, default=10)
    # model
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--dim_hidden', type=int, default=4096)
    parser.add_argument('--dim_out', type=int, default=256)
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=256)
    # tsne
    parser.add_argument('--save_path', type=str, default='tsne_slots.jpg')
    parser.add_argument('--dpi', type=int, default=100)

    args = parser.parse_args()

    # Transforms
    mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])

    # 1) 建立 Dataset
    dataset = ImageFolder(args.dataset, args.data_dir, transform)
    subset_indices = range(5000)  # 或隨機挑 10 個索引
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    # 2) 載入 model
    model = get_model(args).cuda()

    # 3) 取出 Flatten 特徵 & Slots
    all_feats, all_slots, img_idx = get_features_and_assign(model, subset_dataset, args.batch_size)
    print("Features shape:", all_feats.shape)
    print("Slots shape:", all_slots.shape)

    # visualize_tsne(all_feats, all_slots, img_idx, args)
    
    distinct_count_all = count_distinct_slots_per_image(model, dataset, batch_size=256)
    print("Done! distinct_count_all shape:", distinct_count_all.shape)

    plot_distinct_slots(distinct_count_all)

    # Count slots and save to txt
    count_slots_and_save(model, dataset, args.batch_size, 'slot_counts.txt')

    