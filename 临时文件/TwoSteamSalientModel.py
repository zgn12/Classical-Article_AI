import torch.nn as nn

class TwoSteamSalientModel(nn.Module):
    def __init__(self, padding: str = 'same', **kwargs):
        super(TwoSteamSalientModel, self).__init__()

        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        self.activation = nn.__dict__[kwargs['train']['activation']]()
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']

        self.init_model()

    def init_model(self, input: list = None) -> None:
        if input is None:
            input = [
                nn.Input(shape=(1, self.sequence_length * self.sleep_epoch_length, 1), name=f'EEG_input'),
                nn.Input(shape=(1, self.sequence_length * self.sleep_epoch_length, 1), name=f'EOG_input')
            ]
        self.stream1 = self.build_branch(input[0], "EEG")
        self.stream2 = self.build_branch(input[1], "EOG")

        self.mul = nn.Multiply()
        self.merge = nn.Add()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.filters[0], self.filters[0] // 4, kernel_size=1),
            self.activation,
            nn.Conv2d(self.filters[0] // 4, self.filters[0], kernel_size=1),
            nn.Sigmoid()
        )

        self.reshape = nn.Sequential(
            nn.Reshape((self.sequence_length, self.sleep_epoch_length, self.filters[0])),
            nn.Conv2d(self.filters[0], self.filters[0], kernel_size=1, activation='tanh', padding='same'),
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d((1, self.sleep_epoch_length)),
            nn.Conv2d(self.filters[0], 5, kernel_size=(self.kernel_size, 1), padding=self.padding, activation='softmax')
        )

    def forward(self, x):
        stream1 = self.stream1(x[0])
        stream2 = self.stream2(x[1])

        mul = self.mul([stream1, stream2])
        merge = self.merge([stream1, stream2, mul])

        se = self.se(merge)
        x = mul * se

        reshape = self.reshape(x)
        pool = self.pool(reshape)

        return pool

    def build_branch(self, input, pre_name: str = "") -> nn.Module:
        """
        Build one branch of the SalientSleepNet
        """

        class UEncoder(nn.Module):
            def __init__(self, filter_count, depth, middle_layer_filter, kernel_size, pooling_size, padding, activation):
                super().__init__()
                self.filter_count = filter_count
                self.depth = depth
                self.middle_layer_filter = middle_layer_filter
                self.kernel_size = kernel_size
                self.pooling_size = pooling_size
                self.padding = padding
                self.activation = activation

                self.layers = nn.ModuleList([
                    nn.Conv2d(in_channels=1, out_channels=self.filter_count, kernel_size=self.kernel_size, padding=self.padding),
                    self.activation
                ])

                for i in range(self.depth):
                    filter_count = self.filter_count * 2 if i != self.depth - 1 else self.filter_count
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels=self.filter_count, out_channels=filter_count, kernel_size=self.kernel_size,
                                  dilation=2 ** (i + 1), padding=self.padding),
                        self.activation,
                        nn.BatchNorm2d(num_features=filter_count)
                    )
                    self.layers.append(layer)
                    self.filter_count = filter_count

                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.filter_count, out_channels=self.middle_layer_filter, kernel_size=(1, 1),
                              padding=self.padding),
                    self.activation,
                    nn.BatchNorm2d(num_features=self.middle_layer_filter),
                    nn.MaxPool2d(kernel_size=(self.pooling_size, 1))
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                x = self.downsample(x)
                return x

        class MSE(nn.Module):
            def __init__(self, filter_count):
                super().__init__()
                self.filter_count = filter_count
                self.layers = nn.ModuleList([
                    nn.Conv2d(in_channels=self.filter_count, out_channels=self.filter_count, kernel_size=(3, 1),
                            padding='same'),
                    nn.BatchNorm2d(num_features=self.filter_count),
                    self.activation,
                    nn.Conv2d(in_channels=self.filter_count, out_channels=1, kernel_size=(1, 1)),
                    nn.Sigmoid()
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

    encoder = UEncoder(self.filters[0], self.u_depths, self.u_inner_filter, self.kernel_size, self.pooling_sizes[0],
                       self.padding, self.activation)

    mse_layers = []
    for i in range(len(self.mse_filters)):
        mse_layers.append(MSE(self.mse_filters[i]))

    return nn.Sequential(encoder, *mse_layers)


# 训练函数
"""
这个 train_model 函数用于训练模型，并在每个 epoch 结束后输出损失。该函数接受以下参数：

model：要训练的模型
criterion：损失函数
optimizer：优化器
dataloader：数据加载器，返回输入和标签对
num_epochs：训练轮数
predict 函数用于生成预测结果。该函数接受以下参数：

model：已经训练好的模型
dataloader：数据加载器，返回输入数据，不包含标签
"""
import torch

def train_model(model, criterion, optimizer, dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (eeg_input, eog_input, labels) in enumerate(dataloader):
            eeg_input = eeg_input.to(device)
            eog_input = eog_input.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model([eeg_input, eog_input])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * eeg_input.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
def predict(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (eeg_input, eog_input, _) in enumerate(dataloader):
            eeg_input = eeg_input.to(device)
            eog_input = eog_input.to(device)
            outputs = model([eeg_input, eog_input])
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())
            
    return np.concatenate(predictions)
