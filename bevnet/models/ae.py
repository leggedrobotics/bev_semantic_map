import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # (1, 64, 64) -> (32, 16, 16) -> (64, 8, 8) -> (32, 16, 16) -> (1, 64, 64)

        # Encoder
        self.encoder = torch.nn.Sequential(
            # Input size: 1x64x64
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),     # Output size: 16x32x32
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),     # Output size: 32x16x16
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # Output size: 64x8x8
            torch.nn.ReLU(True)
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            # Input size: 64x8x8
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output size: 32x16x16
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # Output size: 16x32x32
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),    # Output size: 1x64x64
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
