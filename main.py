import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import argparse
import datetime
import os
import random
from datasets import FivekDataset
from models import CAN, SandOCAN, UNet
from torch_utils import JoinedDataLoader, load_model
from loss import ColorSSIM, NimaLoss

# Парсим аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--epochs', type=int, default=52, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_idx', type=int, default=0, help='cuda device id')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', type=int, default=10, help='save model every N epochs')
parser.add_argument('--checkpoint_dir', default="checkpoints", help='directory for the checkpoints')
parser.add_argument('--final_dir', default="final_models", help='directory for the final_models')
parser.add_argument('--model_type', default='unet', choices=['can32', 'sandocan32', 'unet'], help='type of model to use')
parser.add_argument('--load_model', action='store_true', help='load from latest checkpoint')
parser.add_argument('--loss', default='mse', choices=['mse', 'mae', 'l1nima', 'l2nima', 'l1ssim', 'colorssim'], help='loss function')
parser.add_argument('--gamma', default=0.001, type=float, help='gamma to be used only in case of Nima Loss')
parser.add_argument('--data_path', default='data', help='path of the dataset')
opt = parser.parse_args()

# Автоматически включаем CUDA, если доступна
opt.cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.cuda_idx}" if opt.cuda else "cpu")

# Устанавливаем seed для воспроизводимости
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

# Создание директорий
os.makedirs(opt.checkpoint_dir, exist_ok=True)
os.makedirs(opt.final_dir, exist_ok=True)

# Логирование в TensorBoard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

# Загрузка датасетов с проверкой
print(f"Loading dataset from {opt.data_path}...")
landscape_transform = transforms.Compose([
    transforms.Resize((332, 500)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Убедимся, что изображение RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
portrait_transform = transforms.Compose([
    transforms.Resize((500, 332)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

landscape_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=landscape_transform, filter_ratio="landscape")
portrait_dataset = FivekDataset(opt.data_path, expert_idx=2, transform=portrait_transform, filter_ratio="portrait")

print(f"Landscape dataset size: {len(landscape_dataset)}, Portrait dataset size: {len(portrait_dataset)}")

# Проверяем, что датасеты не пустые
if len(landscape_dataset) == 0 or len(portrait_dataset) == 0:
    raise ValueError("Error: One of the datasets is empty!")

# Разделение на обучающий и тестовый датасеты
train_size = int(0.8 * len(landscape_dataset))
test_size = len(landscape_dataset) - train_size
train_landscape_dataset, test_landscape_dataset = random_split(landscape_dataset, [train_size, test_size])

train_size = int(0.8 * len(portrait_dataset))
test_size = len(portrait_dataset) - train_size
train_portrait_dataset, test_portrait_dataset = random_split(portrait_dataset, [train_size, test_size])

# Создание DataLoader с num_workers=0, чтобы избежать зависаний
train_landscape_loader = DataLoader(train_landscape_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
train_portrait_loader = DataLoader(train_portrait_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
train_loader = JoinedDataLoader(train_landscape_loader, train_portrait_loader)

test_landscape_loader = DataLoader(test_landscape_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
test_portrait_loader = DataLoader(test_portrait_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
test_loader = JoinedDataLoader(test_landscape_loader, test_portrait_loader)

# Выбор модели
if opt.model_type == 'can32':
    model = CAN(n_channels=32)
elif opt.model_type == 'sandocan32':
    model = SandOCAN()
elif opt.model_type == 'unet':
    model = UNet()
else:
    raise ValueError("Invalid model type!")
model = model.to(device)

# Загрузка модели, если указано
start_epoch = 0
if opt.load_model:
    model, start_epoch = load_model(model, opt.checkpoint_dir, opt.run_tag)

# Выбор функции потерь
loss_functions = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "l1nima": NimaLoss(device, opt.gamma, nn.L1Loss()),
    "l2nima": NimaLoss(device, opt.gamma, nn.MSELoss()),
    "l1ssim": ColorSSIM(device, 'l1'),
    "colorssim": ColorSSIM(device)
}
criterion = loss_functions[opt.loss].to(device)

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# Выбор случайных индексов для тестирования
num_samples = min(3, len(test_landscape_dataset))
test_idxs = random.sample(range(len(test_landscape_dataset)), num_samples)

# Основной цикл обучения
for epoch in range(start_epoch, opt.epochs):
    model.train()
    cumulative_loss = 0.0
    for i, (im_o, im_t) in enumerate(train_loader):
        im_o, im_t = im_o.to(device), im_t.to(device)
        optimizer.zero_grad()
        output = model(im_o)
        loss = criterion(output, im_t)
        loss.backward()
        optimizer.step()
        cumulative_loss += loss.item()
    print(f"Epoch {epoch+1}/{opt.epochs}, Avg Loss: {cumulative_loss / len(train_loader):.4f}")

    # Сохранение модели каждые `checkpoint_every` эпох
    if (epoch + 1) % opt.checkpoint_every == 0:
        torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, f"{opt.run_tag}_epoch{epoch+1}.pt"))

print("Training completed successfully!")
torch.save(model.state_dict(), os.path.join(opt.final_dir, f"{opt.run_tag}_final.pt"))
