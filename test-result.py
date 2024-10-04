# test-result.py
import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from ttt_model_2d_4 import TTTModel2D # 对应ttt_model_2d_1.py FCN+BLOCK2D
from ttt_config import TTTConfig
from data_loading import NPY_datasets
from loss_functions import BceDiceLoss
from metrics import compute_metrics
from tqdm import tqdm
from thop import profile
from datetime import datetime
from sklearn.metrics import confusion_matrix
def get_scheduler(config, optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.000001,
            last_epoch = -1
        )
    return scheduler
# Configuration
config = TTTConfig(
    hidden_size=768,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    hidden_act="silu",
    img_size=256,
    patch_size=8,
    in_channels=3,
    num_classes=1,  # Binary segmentation
)

# Dataset and DataLoader
train_dataset = NPY_datasets('./isic2017/', config, train=True)
val_dataset = NPY_datasets('./isic2017/', config, train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Model
model = TTTModel2D(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = BceDiceLoss()
optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 0.001,
            betas = (0.9, 0.999),
            eps = 1e-8 ,
            weight_decay = 0.0001,
            amsgrad = False
        )
scheduler = get_scheduler(config, optimizer)

# 获取当前时间，格式化为子文件夹名
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join('result/test', start_time)

# 创建以当前时间命名的子文件夹
os.makedirs(result_dir, exist_ok=True)
# 保存模型路径
#best_model_path = os.path.join(result_dir, 'best_model.pth')

if os.path.exists("result/20241003-033442-patch-8-best1-isic18/best_model.pth"):
    # 加载模型权重
    weight_dict = torch.load("result/20241003-033442-patch-8-best1-isic18/best_model.pth", map_location=device)

    # 移除不需要的键
    keys_to_remove = [key for key in weight_dict.keys() if "total_ops" in key or "total_params" in key]
    for key in keys_to_remove:
        del weight_dict[key]

    # 加载清理后的权重
    model.load_state_dict(weight_dict)
    print('Successfully loading checkpoint.')


log_file_path = os.path.join(result_dir, 'training_log.txt')

preds_list=[]
gts_list=[]

# 打开日志文件
with open(log_file_path, 'w') as log_file:
    
    model.eval()
    metrics = {'accuracy': 0, 'sensitivity': 0, 'specificity': 0, 'f1_score': 0, 'iou': 0}
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            gts_list.append(masks.squeeze(1).cpu().detach().numpy())
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            batch_metrics = compute_metrics(preds, masks)
            for key in metrics.keys():
                metrics[key] += batch_metrics[key]
            val_loss += loss.item()
            if type(outputs) is tuple:
                outputs = outputs[0]
            outputs = outputs.squeeze(1).cpu().detach().numpy()
            preds_list.append(outputs)
    preds_list = np.array(preds_list).reshape(-1)
    gts_list = np.array(gts_list).reshape(-1)
    for key in metrics.keys():
        metrics[key] /= len(val_loader)
    threshold = 0.5
    y_pre = np.where(preds_list>=threshold, 1, 0)
    y_true = np.where(gts_list>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0   
    log_info = f'loss: {val_loss / len(val_loader):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    print(log_info)
    #val_log = f"test_loss: {val_loss / len(val_loader):.4f}, Validation Metrics: {metrics}"
    #print(val_log)
    log_file.write(log_info + "\n")