import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

class STNNet(nn.Module):
    """
    Spatial Transformer Network (STN) model for image transformation prior to classification.
    This model uses a localization network and affine transformation to align images spatially 
    before passing them to convolutional layers for classification.

    Attributes:
        localization (nn.Sequential): A localization network for affine transformations.
        fc_loc (nn.Sequential): Fully connected layers to regress affine transformation parameters.
    """
    def __init__(self):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize weights with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the spatial transformer network to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor with image data.
        
        Returns:
            torch.Tensor: Transformed tensor.
        """
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STN model.
        
        Args:
            x (torch.Tensor): Input tensor with image data.
        
        Returns:
            torch.Tensor: Output tensor with classification logits.
        """
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SuperResolutionNet(nn.Module):
    """
    A Super-Resolution Convolutional Neural Network (SRCNN) that increases
    image resolution by an upscale factor using a sub-pixel convolution layer.

    This model uses the efficient sub-pixel convolution layer described in
    "Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network" - Shi et al <https://arxiv.org/abs/1609.05158>
    for increasing the resolution of an image by an upscale factor.
    The model expects the Y component of the ``YCbCr`` of an image as an input, and
    outputs the upscaled Y component in super resolution.
    
    Args:
        upscale_factor (int): Factor by which to increase resolution.
        inplace (bool): Whether to perform in-place activation.
    """
    def __init__(self, upscale_factor: int, inplace: bool = False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SRCNN.
        
        Args:
            x (torch.Tensor): Input tensor with low-resolution image data.
        
        Returns:
            torch.Tensor: Output tensor with super-resolved image data.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        """Initializes weights of convolutional layers using orthogonal initialization."""
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

class ClassificationNet(nn.Module):
    def __init__(self, input_channels=3, input_size=(32, 32), hidden_conv_dims=None, hidden_fc_dims=None, num_classes: int = 10):
        """
        Flexible classification network with variable convolutional layers based on hidden_conv_dims.

        Args:
            input_channels (int): Number of channels in the input image (default is 3 for RGB).
            input_size (tuple of int): Height and width of the input image (default is (32, 32)).
            hidden_conv_dims (list of int): List where each entry specifies the number of output channels
                                       for a convolutional layer. If empty, no convolutional layers
                                       are created.
            hidden_fc_dims (list of int): List where each entry specifies the number of output channels
                                       for a fully connected layer. If empty, no layers are created.
            num_classes (int): Number of classes in the output.
        """
        super().__init__()

        # Initialize convolutional layers
        self.conv_layers = nn.Sequential() if not hidden_conv_dims else self._build_conv_layers(input_channels, hidden_conv_dims)

        # Calculate input size for the fully connected layer
        fc_input_dim = self._get_conv_output_dim(input_size, input_channels) if hidden_conv_dims else input_channels * input_size[0] * input_size[1]
        
        # Initialize convolutional layers
        self.fc_layers = nn.Sequential() if not hidden_fc_dims else self._build_fc_layers(fc_input_dim, hidden_fc_dims)

        # Calculate input size for the last layer
        final_input_dim = hidden_fc_dims[-1] if hidden_fc_dims else fc_input_dim
        self.fc_layers.append( nn.Linear(final_input_dim, num_classes) )

    def _build_conv_layers(self, input_channels, hidden_dims):
        layers = []
        in_channels = input_channels
        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=0))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_fc_layers(self, input_size, hidden_dims):
        layers = []
        in_size = input_size
        for out_size in hidden_dims:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        return nn.Sequential(*layers)

    def _get_conv_output_dim(self, input_size, input_channels):
        """
        Calculates the output dimension of the convolutional layers.

        Args:
            input_size (tuple): The height and width of the input image.
            input_channels (int): The number of input channels.

        Returns:
            int: Flattened size of the output from the convolutional layers.
        """
        if not self.conv_layers:
            # No convolutional layers, return flat input size
            return input_channels * input_size[0] * input_size[1]
        
        # Create a dummy input tensor with the specified input size
        dummy_input = torch.zeros(1, input_channels, *input_size)
        output = self.conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))

    def forward(self, x):
        # Apply convolutional layers if any
        if len(self.conv_layers) > 0:
            x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

class LeNet(nn.Module):
    """
    A CNN-based model loosely based on LeNet architecture, designed for classification tasks.
    
    Args:
        img_size (int): Dimension of the input image size.
    """
    def __init__(self, img_size: int = 6):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * img_size * img_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LeNet model.
        
        Args:
            x (torch.Tensor): Input tensor with image data.
        
        Returns:
            torch.Tensor: Output tensor with classification logits.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x: torch.Tensor) -> int:
        """
        Calculates the number of features in a tensor, excluding the batch dimension.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            int: Number of features.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
