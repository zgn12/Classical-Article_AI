import torch
import torch.nn as nn

class SingleSalientModel(nn.Module):
    def __init__(self, padding: str ='same', **kwargs):
        super(SingleSalientModel, self).__init__()
        
        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        self.activation = kwargs['train']['activation']
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']
        
        # Model
        self.model = nn.Sequential(*self.init_model())
    
    def init_model(self) -> list:
        model = []
        
        l_name = "single_model_enc"
        
        # encoder 1
        u1 = create_u_encoder(nn.Sequential(), self.filters[0], self.kernel_size,
                              self.pooling_sizes[0], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[0], pre_name=l_name, idx=1, padding=self.padding,
                              activation=self.activation)
        u1.add_module(f"{l_name}_reduce_dim_layer_1", nn.Conv2d(u1[-1].out_channels, int(u1[-1].out_channels * 0.5),
                                                                  kernel_size=(1, 1), padding=self.padding))
        u1.add_module(f"{l_name}_pool1", nn.MaxPool2d((self.pooling_sizes[0], 1)))
        model.extend(u1)
        
        # encoder 2
        u2 = create_u_encoder(nn.Sequential(), self.filters[1], self.kernel_size,
                              self.pooling_sizes[1], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[1], pre_name=l_name, idx=2, padding=self.padding,
                              activation=self.activation)
        u2.add_module(f"{l_name}_reduce_dim_layer_2", nn.Conv2d(u2[-1].out_channels, int(u2[-1].out_channels * 0.5),
                                                                  kernel_size=(1, 1), padding=self.padding))
        u2.add_module(f"{l_name}_pool2", nn.MaxPool2d((self.pooling_sizes[1], 1)))
        model.extend(u2)
        
        # encoder 3
        u3 = create_u_encoder(nn.Sequential(), self.filters[2], self.kernel_size,
                              self.pooling_sizes[2], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[2], pre_name=l_name, idx=3, padding=self.padding,
                              activation=self.activation)
        u3.add_module(f"{l_name}_reduce_dim_layer_3", nn.Conv2d(u3[-1].out_channels, int(u3[-1].out_channels * 0.5),
                                                                  kernel_size=(1, 1), padding=self.padding))
        u3.add_module(f"{l_name}_pool3", nn.MaxPool2d((self.pooling_sizes[2], 1)))
        model.extend(u3)
        
        # encoder 4
        u4 = create_u_encoder(nn.Sequential(), self.filters[3], self.kernel_size,
                              self.pooling_sizes[3], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[3], pre_name=l_name, idx=4, padding=self.padding,
                              activation=self.activation)
        u4.add_module(f"{l_name}_reduce_dim_layer_4", nn.Conv2d(u4[-1].out_channels, int(u4[-1].out_channels * 0.5),
                                                                  kernel_size=(1, 1), padding=self.padding))
        u4.add_module(f"{l_name}_pool4", nn.MaxPool2d((self.pooling_sizes[3], 1)))
        model.extend(u4)
        
        # encoder 5
        u5 = create_u_encoder(nn.Sequential(), self.filters[4], self.kernel_size,
                              self.pooling_sizes[3], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[3], pre_name=l_name, idx=5, padding=self.padding,
                              activation=self.activation)
        u5.add_module(f"{l_name}_reduce_dim_layer_5", nn.Conv2d(u5[-1].out_channels, int(u5[-1].out_channels * 0.5),
                                                                  kernel_size=(1, 1), padding=self.padding))
        u5.add_module(f"{l_name}_pool5", nn.MaxPool2d((self.pooling_sizes[3], 1)))
        model.extend(u5)

        # bridge
        b = create_bridge(nn.Sequential(), self.filters[5], self.kernel_size, self.dilation_sizes,
                        padding=self.padding, activation=self.activation)
        model.extend(b)
        
        # decoder 1
        d1 = create_decoder(nn.Sequential(), self.filters[4], self.kernel_size, self.dilation_sizes,
                            middle_layer_filter=self.u_inner_filter, depth=self.u_depths[3],
                            pre_name=l_name, idx=5, padding=self.padding, activation=self.activation)
        d1.add_module(f"{l_name}_up1", nn.Upsample(scale_factor=(self.pooling_sizes[3], 1)))
        model.extend(d1)
        
        # decoder 2
        d2 = create_decoder(nn.Sequential(), self.filters[3], self.kernel_size, self.dilation_sizes,
                            middle_layer_filter=self.u_inner_filter, depth=self.u_depths[2],
                            pre_name=l_name, idx=4, padding=self.padding, activation=self.activation)
        d2.add_module(f"{l_name}_up2", nn.Upsample(scale_factor=(self.pooling_sizes[3]*self.pooling_sizes[2], 1)))
        model.extend(d2)
        
        # decoder 3
        d3 = create_decoder(nn.Sequential(), self.filters[2], self.kernel_size, self.dilation_sizes,
                            middle_layer_filter=self.u_inner_filter, depth=self.u_depths[1],
                            pre_name=l_name, idx=3, padding=self.padding, activation=self.activation)
        d3.add_module(f"{l_name}_up3", nn.Upsample(scale_factor=(self.pooling_sizes[3]*self.pooling_sizes[2]*self.pooling_sizes[1], 1)))
        model.extend(d3)
        
        # decoder 4
        d4 = create_decoder(nn.Sequential(), self.filters[1], self.kernel_size, self.dilation_sizes,
                            middle_layer_filter=self.u_inner_filter, depth=self.u_depths[0],
                            pre_name=l_name, idx=2, padding=self.padding, activation=self.activation)
        d4.add_module(f"{l_name}_up4", nn.Upsample(scale_factor=(self.pooling_sizes[3]*self.pooling_sizes[2]*self.pooling_sizes[1]*self.pooling_sizes[0], 1)))
        model.extend(d4)
        
        # decoder 5
        d5 = create_decoder(nn.Sequential(), self.filters[0], self.kernel_size, self.dilation_sizes,
                            middle_layer_filter=self.u_inner_filter, depth=self.u_depths[0],
                            pre_name=l_name, idx=1, padding=self.padding, activation=self.activation)
        d5.add_module(f"{l_name}_up5", nn.Upsample(scale_factor=(self.sequence_length, 1)))
        model.extend(d5)
        
        # final layer
        model.append(nn.Conv2d(self.filters[0], 1, kernel_size=(1, 1), padding=self.padding))
        
        return model

    def forward(self, x):
        return self.model(x)
    


# Helper functions
def create_u_encoder(model, filter_count: int, kernel_size: tuple, pool_size: int, middle_layer_filter: int,
depth: int, pre_name: str, idx: int, padding: str ='same',
activation=nn.ReLU(inplace=True)) -> nn.Sequential:
    """
    Create a U-Net encoder component.

    Parameters:
    model (nn.Sequential): The initial model to append the encoder to.
    filter_count (int): The number of filters to use.
    kernel_size (tuple): The size of the convolutional kernel, e.g. (3, 3).
    pool_size (int): The size of the pooling window.
    middle_layer_filter (int): The number of filters to use in the middle layer.
    depth (int): The depth of the encoder component.
    pre_name (str): The prefix name of the layers.
    idx (int): The index of the U-Net block.
    padding (str): The type of padding to use ('same' or 'valid').
    activation (nn.Module): The activation function to use.

    Returns:
    nn.Sequential: The final model.
    """
    for i in range(depth):
        if i == 0:
            conv_layer = nn.Conv2d(1, filter_count, kernel_size=kernel_size, padding=padding)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_conv", conv_layer)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_act", activation)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_batch", nn.BatchNorm2d(filter_count))
            
        else:
            conv_layer = nn.Conv2d(filter_count, filter_count*2, kernel_size=kernel_size, padding=padding)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_conv0", conv_layer)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_act0", activation)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_batch0", nn.BatchNorm2d(filter_count*2))

            conv_layer = nn.Conv2d(filter_count*2, filter_count, kernel_size=kernel_size, padding=padding)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_conv1", conv_layer)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_act1", activation)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_batch1", nn.BatchNorm2d(filter_count))
            
        if i < depth - 1:
            pool_layer = nn.MaxPool2d(pool_size, stride=pool_size)
            model.add_module(f"{pre_name}_u{idx}_{i+1}_pool", pool_layer)
            filter_count *= 2
            
    return model


def create_bridge(model, filter_count: int, kernel_size: tuple, dilation_sizes: list,
padding: str ='same', activation=nn.ReLU(inplace=True)) -> nn.Sequential:
    """
    Create the bridge component for the U-Net.

    Parameters:
    model (nn.Sequential): The initial model to append the bridge to.
    filter_count (int): The number of filters to use.
    kernel_size (tuple): The size of the convolutional kernel, e.g. (3, 3).
    dilation_sizes (list): A list of the dilation sizes to use for each convolutional layer.
    padding (str): The type of padding to use ('same' or 'valid').
    activation (nn.Module): The activation function to use.

    Returns:
    nn.Sequential: The final model.
    """
    for i, dilation in enumerate(dilation_sizes):
        conv_layer = nn.Conv2d(filter_count, filter_count, kernel_size=kernel_size, dilation=dilation, padding=padding)
        model.add_module(f"bridge_{i+1}_conv", conv_layer)
        model.add_module(f"bridge_{i+1}_act", activation)
        model.add_module(f"bridge_{i+1}_batch", nn.BatchNorm2d(filter_count))
        
    return model


def create_decoder(model, filter_count: int, kernel_size: tuple, dilation_sizes: list,
middle_layer_filter: int, depth: int, pre_name: str, idx: int, padding: str ='same',
activation=nn.ReLU(inplace=True)) -> nn.Sequential:
    """
    Create a U-Net decoder component.

    Parameters:
    model (nn.Sequential): The initial model to append the decoder to.
    filter_count (int): The number of filters to use.
    kernel_size (tuple): The size of the convolutional kernel, e.g. (3, 3).
    dilation_sizes (list): A list of the dilation sizes to use for each convolutional layer.
    middle_layer_filter (int): The number of filters to use in the middle layer.
    depth (int): The depth of the decoder component.
    pre_name (str): The prefix name of the layers.
    idx (int): The index of the U-Net block.
    padding (str): The type of padding to use ('same' or 'valid').
    activation (nn.Module): The activation function to use.

    Returns:
    nn.Sequential: The final model.
    """
    for i in reversed(range(depth)):
        if i == depth-1:
            conv_layer = nn.ConvTranspose2d(filter_count, middle_layer_filter, kernel_size=kernel_size, stride=(2, 1), padding=padding)
        else:
            conv_layer = nn.ConvTranspose2d(filter_count*2, filter_count, kernel_size=kernel_size, stride=(2, 1), padding=padding)
            
        model.add_module(f"{pre_name}_d{idx}_{i+1}_conv", conv_layer)
        model.add_module(f"{pre_name}_d{idx}_{i+1}_act", activation)
        model.add_module(f"{pre_name}_d{idx}_{i+1}_batch", nn.BatchNorm2d(filter_count))
        
        if i < depth - 1:
            dilate_layer = nn.Conv2d(filter_count, filter_count, kernel_size=kernel_size, dilation=dilation_sizes[i], padding=padding)

        model.add_module(f"{pre_name}_d{idx}_{i+1}_dilate", dilate_layer)
        model.add_module(f"{pre_name}_d{idx}_{i+1}_act_dilate", activation)
        model.add_module(f"{pre_name}_d{idx}_{i+1}_batch_dilate", nn.BatchNorm2d(filter_count))
    
    filter_count //= 2
    
    return model


# Example Usage
if name == "main":
    # Define Hyperparameters
    filters = [32, 64, 128, 256, 512]
    kernel_size = (3, 3)
    pooling_sizes = [2, 2, 2, 2]
    dilation_sizes = [1, 2, 4, 8]
    u_depths = [4, 3, 2, 1]
    u_inner_filter = 1024
    sequence_length = 100

    # Initialize the Model
    model = UNet(filters, kernel_size, pooling_sizes, dilation_sizes, u_depths, 
                u_inner_filter, sequence_length)

    # Define Input Shape
    input_shape = (1, sequence_length, 12)
    x = torch.randn(input_shape)

    # Print Model Output Shape
    print(model(x).shape)