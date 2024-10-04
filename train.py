import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from ttt_model_2d import TTTModel2D # 对应ttt_model_2d_1.py FCN+BLOCK2D
from ttt_config import TTTConfig
from data_loading import NPY_datasets
from loss_functions import BceDiceLoss
from metrics import compute_metrics
from tqdm import tqdm
from thop import profile
from datetime import datetime

def get_scheduler(config, optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.000001,
            last_epoch = -1
        )
    return scheduler
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

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
train_dataset = NPY_datasets('./isic2018/', config, train=True)
val_dataset = NPY_datasets('./isic2018/', config, train=False)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

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


def cal_params_flops(model, size):
    input = torch.randn(1, 3, size, size).to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops', flops / 1e9)  # 打印计算量
    print('params', params / 1e6)  # 打印参数量
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
cal_params_flops(model,256)
# 获取当前时间，格式化为子文件夹名
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join('result', start_time)

# 创建以当前时间命名的子文件夹
os.makedirs(result_dir, exist_ok=True)

# 保存模型路径
best_model_path = os.path.join(result_dir, 'best_model.pth')

# Training loop
num_epochs = 150
best_val_loss = float('0.6')

scheduler = cosine_scheduler(base_value=0.002,final_value=0.000001,epochs=num_epochs,
                                 niter_per_ep=len(train_loader),warmup_epochs=10,start_warmup_value=5e-4)

if os.path.exists("result/20241002-150210/best_model.pth"):
        weight_dict = torch.load("result/20241002-150210/best_model.pth", map_location=device)
        model.load_state_dict(weight_dict)
        print('Successfully loading checkpoint.')

log_file_path = os.path.join(result_dir, 'training_log.txt')

# 打开日志文件
with open(log_file_path, 'w') as log_file:
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader)
        for it, (images, masks) in enumerate(pbar):
            it = len(train_loader) * epoch + it
            param_group = optimizer.param_groups[0]
            param_group['lr'] = scheduler[it]
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # Adjust shape if necessary
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 记录训练日志
        epoch_log = f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
        print(epoch_log)
        log_file.write(epoch_log + "\n")



        # Validation
        model.eval()
        metrics = {'accuracy': 0, 'sensitivity': 0, 'specificity': 0, 'f1_score': 0,'iou': 0}
        val_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                outputs = outputs.squeeze(1)
                loss_val = criterion(outputs, masks)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                masks = (masks > 0.5).float()
                batch_metrics = compute_metrics(preds, masks)
                for key in metrics.keys():
                    metrics[key] += batch_metrics[key]
                val_loss += loss_val.item()

        for key in metrics.keys():
            metrics[key] /= len(val_loader)
        
        val_log = f"Val_loss: {val_loss / len(val_loader):.4f}, Validation Metrics: {metrics}"
        print(val_log)
        log_file.write(val_log + "\n")

        # 保存最佳模型
        val_loss = val_loss / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val_loss: {best_val_loss:.4f}")
            log_file.write(f"Best model saved with val_loss: {best_val_loss:.4f}\n")
