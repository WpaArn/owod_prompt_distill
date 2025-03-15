import clip
import torch
from pascal_voc import VOC_COCO_CLASS_NAMES

import os
import clip
import torch

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
class_names = VOC_COCO_CLASS_NAMES["IOD"] 

# 创建保存特征的目录
output_dir = "datasets/clip/IOD"
os.makedirs(output_dir, exist_ok=True)

# 逐个类别生成文本特征并保存到对应的 .txt 文件中
i=0
for class_name in class_names:
    # 生成文本特征
    if class_name == "unknown":
        text_prompt = "an unknown object of uncertain type"
    else:
        text_prompt = f"a photo of a {class_name}"
    i=i+1
    text_inputs = clip.tokenize([text_prompt]).to(device)
    # 使用 CLIP 文本编码器生成文本特征
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # 对文本特征进行归一化
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 将文本特征转换为 NumPy 数组
    text_features_np = text_features.cpu().numpy()

    # 保存文本特征到 txt 文件
    output_file_path = os.path.join(output_dir, f"{class_name}.txt")
    with open(output_file_path, "w") as f:
        # 将特征向量中的元素转换为字符串并以空格分隔
        feature_str = " ".join(map(str, text_features_np.flatten()))
        f.write(feature_str + "\n")
print(i)
print("所有类别的文本特征已成功保存到目录下。")


# if __name__ == "__main__":
#     # 示例用法
#     encoder = CLIPTextEncoder()
#     class_names = ["car", "dog", "tree", "person" , "unknown object"]
#     text_features = encoder.encode_text_prompts(class_names)
#     print(text_features.shape)  # 输出: torch.Size([4, 512]) 