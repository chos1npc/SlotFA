import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
def plot_coco_val_object_distribution(ann_file, images_dir):
    """
    讀取 COCO val 的 annotation JSON，計算每張圖有多少 objects，並畫出分佈圖。
    ann_file: 例如 'instances_val2017.json'
    images_dir: 圖片目錄，用於補齊未在 JSON 中標註的圖片。
    """
    # 1. 讀取 json
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)

    # 2. 計算每張圖的 object 數量
    image_obj_count = defaultdict(int)  # key: file_name, val: 計數

    for ann in ann_data["annotations"]:
        image_id = ann["image_id"]
        file_name = next(image["file_name"] for image in ann_data["images"] if image["id"] == image_id)
        image_obj_count[file_name] += 1

    # 從圖片目錄中提取所有的圖片文件名
    dir_image_files = {file_name for file_name in os.listdir(images_dir) if file_name.endswith('.jpg')}

    # 找出 JSON 中未標註的圖片，補齊 object 數為 0
    for file_name in dir_image_files:
        if file_name not in image_obj_count:
            image_obj_count[file_name] = 0

    # 確保包含所有圖片（JSON + 目錄中的圖片）
    all_images = dir_image_files.union(image_obj_count.keys())
    counts = [image_obj_count[file_name] for file_name in sorted(all_images)]

    mean_count = np.mean(counts)
    median_count = np.median(counts)
    print(f"Average object count: {mean_count:.2f}")
    print(f"Median object count: {median_count:.2f}")

    # 打印圖片總數
    print(f"Total number of images: {len(all_images)}")

    # 保存每張圖片的物件數量到 txt 文件
    output_file = "image_object_count.txt"
    with open(output_file, 'w') as f:
        for count in counts:
            f.write(f"{count}\n")
    print(f"Image object counts saved to {output_file}")

    # 畫分佈圖 (直方圖)
    plt.figure(figsize=(6, 4))
    max_count = max(counts)
    # bins 設定為 0~(max_count+1) 每個整數區間一個 bin
    plt.hist(counts, bins=range(max_count + 2), color='skyblue', edgecolor='black')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(range(0, max_count + 1, 2))

    plt.xlabel("Number of objects in an image")
    plt.ylabel("Number of images")
    plt.title("COCO val Object Distribution")
    plt.tight_layout()
    plt.show()

def find_missing_images(json_path, images_dir):
    """
    找出目錄中存在但 JSON 中未包含的圖片。
    
    json_path: COCO annotation JSON 文件的路徑。
    images_dir: 圖片目錄。
    """
    # 讀取 COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 從 JSON 中提取所有的圖片文件名
    json_image_files = {image['file_name'] for image in coco_data['images']}

    # 從圖片目錄中提取所有的圖片文件名
    dir_image_files = {file_name for file_name in os.listdir(images_dir) if file_name.endswith('.jpg')}

    # 找出目錄中存在但不在 JSON 中的圖片
    missing_images = dir_image_files - json_image_files

    # 打印結果
    print(f"Total images in directory: {len(dir_image_files)}")
    print(f"Total images in JSON: {len(json_image_files)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_images:
        print("List of missing images:")
        for img in sorted(missing_images):
            print(img)
    else:
        print("No missing images found.")
if __name__ == "__main__":
    # 替換成你的 COCO val JSON 檔路徑
    ann_file_path = "/media/mmlab206/18b85164-d1ff-4800-80c0-08899eb52cae1/yonghonglin/coco2017/annotations/instances_val2017.json"
    images_dir = "/media/mmlab206/18b85164-d1ff-4800-80c0-08899eb52cae1/yonghonglin/coco2017/val2017"
    plot_coco_val_object_distribution(ann_file_path, images_dir)
    find_missing_images(ann_file_path, images_dir)

