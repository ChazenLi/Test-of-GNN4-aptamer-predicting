import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from gnn_model import GNNModel  # 导入模型

# 加载数据
def load_data(data_dir):
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.pt'):
            try:
                data = torch.load(os.path.join(data_dir, file))
                data_list.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    return data_list

# 训练模型
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 测试模型
def test(model, loader, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = F.mse_loss(output, data.pos)
            total_loss += loss.item()
            mae = mean_absolute_error(output.cpu().detach().numpy(), data.pos.cpu().numpy())
            total_mae += mae
    return total_loss / len(loader), total_mae / len(loader)

def main():
    data_dir = "E:/APTAMER-GEN/pt"
    data_list = load_data(data_dir)
    
    # 检查是否加载了数据
    if not data_list:
        print("No data found in the specified directory.")
        return
    
    print(f"Loaded {len(data_list)} data files.")
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(data_list, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(input_dim=3, hidden_dim=128, output_dim=3, dropout_rate=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    writer = SummaryWriter()
    
    epochs = 200
    best_test_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_mae = test(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('MAE/test', test_mae, epoch)
        
        scheduler.step()
        
        # 保存模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    writer.close()

if __name__ == "__main__":
    main()