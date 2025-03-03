import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention module from SWINN transformer"""
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        self.norm = nn.LayerNorm([in_channels])
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        orig_x = x
        
        # Compute query, key, value
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, D*H*W).permute(0, 1, 3, 2)  # B, heads, D*H*W, head_dim
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, D*H*W)  # B, heads, head_dim, D*H*W
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, D*H*W).permute(0, 1, 3, 2)  # B, heads, D*H*W, head_dim
        
        # Compute attention scores
        attention = torch.matmul(q, k) / (self.head_dim ** 0.5)  # B, heads, D*H*W, D*H*W
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.matmul(attention, v)  # B, heads, D*H*W, head_dim
        out = out.permute(0, 1, 3, 2).reshape(batch_size, C, D, H, W)
        out = self.out(out)
        
        # Add residual connection
        out = out + orig_x
        
        # Apply normalization (adapting LayerNorm to 3D)
        out = out.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network from transformer"""
    def __init__(self, in_channels, expansion_factor=4):
        super(FeedForward, self).__init__()
        hidden_features = in_channels * expansion_factor
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_features, in_channels, kernel_size=1)
        )
        self.norm = nn.LayerNorm([in_channels])
        
    def forward(self, x):
        orig_x = x
        x = self.net(x)
        x = x + orig_x
        
        # Apply normalization (adapting LayerNorm to 3D)
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        
        return x


class SWINNTransformerBlock(nn.Module):
    """SWINN Transformer block"""
    def __init__(self, in_channels, num_heads=8):
        super(SWINNTransformerBlock, self).__init__()
        self.attention = SelfAttention(in_channels, num_heads)
        self.feedforward = FeedForward(in_channels)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class VNetDownBlock(nn.Module):
    """Downsampling block for V-Net"""
    def __init__(self, in_channels, out_channels, num_conv=2):
        super(VNetDownBlock, self).__init__()
        
        # First, downsample the input
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Then apply convolutions
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
            layers.append(nn.PReLU())
        
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.down(x)
        x = self.conv_block(x)
        return x


class VNetUpBlock(nn.Module):
    """Upsampling block for V-Net"""
    def __init__(self, in_channels, out_channels, num_conv=2):
        super(VNetUpBlock, self).__init__()
        
        # Upsampling operation
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Convolution operations after concatenation with skip connection
        # After concatenation, input channels will be out_channels + skip_channels
        layers = []
        # First layer takes concatenated features
        layers.append(nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
        layers.append(nn.PReLU())
        
        # Subsequent layers
        for _ in range(num_conv - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
            layers.append(nn.PReLU())
        
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Make sure skip connection has same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class InputBlock(nn.Module):
    """Initial convolutional block"""
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.PReLU()
        )
        
    def forward(self, x):
        return self.conv(x)


class OutputBlock(nn.Module):
    """Final convolutional layer to produce segmentation"""
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class SWINNVNet(nn.Module):
    """Combined SWINN Transformer + V-Net architecture for BraTS brain tumor segmentation"""
    def __init__(self, in_channels=4, out_channels=3, init_features=16, transformer_blocks=2):
        super(SWINNVNet, self).__init__()
        
        features = init_features
        
        # Initial block
        self.input_block = InputBlock(in_channels, features)
        
        # Encoder pathway with clear channel progression
        self.down1 = VNetDownBlock(features, features*2)
        self.down2 = VNetDownBlock(features*2, features*4)
        self.down3 = VNetDownBlock(features*4, features*8)
        
        # Bottleneck with SWINN Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(transformer_blocks):
            self.transformer_blocks.append(SWINNTransformerBlock(features*8, num_heads=8))
            
        # Decoder pathway
        self.up1 = VNetUpBlock(features*8, features*4)
        self.up2 = VNetUpBlock(features*4, features*2)
        self.up3 = VNetUpBlock(features*2, features)
        
        # Output segmentation layer
        self.output_block = OutputBlock(features, out_channels)
        
    def forward(self, x):
        # Print input shape for debugging
        # print(f"Input shape: {x.shape}")
        
        # Encoder
        x1 = self.input_block(x)
        # print(f"After input block: {x1.shape}")
        
        x2 = self.down1(x1)
        # print(f"After down1: {x2.shape}")
        
        x3 = self.down2(x2)
        # print(f"After down2: {x3.shape}")
        
        x4 = self.down3(x3)
        # print(f"After down3: {x4.shape}")
        
        # Transformer bottleneck
        for transformer in self.transformer_blocks:
            x4 = transformer(x4)
        # print(f"After transformer: {x4.shape}")
            
        # Decoder
        x = self.up1(x4, x3)
        # print(f"After up1: {x.shape}")
        
        x = self.up2(x, x2)
        # print(f"After up2: {x.shape}")
        
        x = self.up3(x, x1)
        # print(f"After up3: {x.shape}")
        
        # Output
        out = self.output_block(x)
        # print(f"Output shape: {out.shape}")
        
        return out


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation"""
    def __init__(self, alpha=0.5, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        dice_loss = self.dice_loss(torch.sigmoid(predictions), targets)
        bce_loss = self.bce_loss(predictions, targets)
        total_loss = self.alpha * dice_loss + (1 - self.alpha) * bce_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(predictions, targets)
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice, BCE and Focal loss for robust segmentation"""
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, predictions, targets):
        sigmoid_preds = torch.sigmoid(predictions)
        
        dice = self.dice_loss(sigmoid_preds, targets)
        bce = self.bce_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal