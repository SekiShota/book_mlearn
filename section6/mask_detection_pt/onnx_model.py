# モデルのアーキテクチャ定義
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,16,5)
        self.fc1=nn.Linear(9*9*16,256)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(256,2)
        
    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1, 9*9*16)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
    
net=Net()
print(net)     


# pthで保存したモデルのパラメータ読み込み
import torch

load_path="./models/pytorch_mask.pth"
load_weight=torch.load(load_path)
net.load_state_dict(load_weight)

# onnxファイル形式で保存
# Input to the model
x = torch.randn(1, 3, 50, 50)
torch_out = net(x)

    # Export the model as onnx (lenet5.onnx)
torch.onnx.export(net,             # model being run
                x,               # model input (or a tuple for multiple inputs)
                "mask_detect.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                'output' : {0 : 'batch_size'}}
                )