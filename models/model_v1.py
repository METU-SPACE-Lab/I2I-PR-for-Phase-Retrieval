import torch

from models import register_model


class End2End(torch.nn.Module):
    def __init__(
        self,
        T,
        K,
        mask_slice,
        mask,
        normalization,
        isShared=False,
        isDnCNN=True,
        nLayersDnCNN=17,
    ):  # T: how many steps
        super().__init__()

        if isShared:
            self.denoisers = DnCNN(num_layers=nLayersDnCNN) if isDnCNN else FBPCONVNet()
        else:
            self.denoisers = torch.nn.ModuleList(
                [
                    (DnCNN(num_layers=nLayersDnCNN) if isDnCNN else FBPCONVNet())
                    for i in range(T)
                ]
            )

        # lam 0 1 arası olmalı!
        # self.lam = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(2.0), requires_grad=True) for i in range(T)]) #sigmoid
        # self.lam = torch.logspace(-.1, -.8, T * 7) # 1....0.125 tezden
        self.lam = torch.logspace(-0.1, -0.8, T)  # 1....0.125 tezden

        self.T = T
        self.K = K

        # self.beta = torch.nn.Parameter(torch.tensor(0.9))
        self.beta = 0.9

        self.mask_slice = mask_slice
        self.mask = mask
        self.normalization = normalization
        self.isShared = isShared

    def forward(self, x, b):
        x_ = x

        # for i in range(self.T * 7):
        for i in range(self.T):
            if self.isShared:
                z_ = self.denoisers(x_)
            else:
                z_ = self.denoisers[i](x_)
                # z_ = self.denoisers[i // 7](x_)

            # x_ = self.dc(z_, b, i // 7)
            x_ = self.dc(z_, b, i)

        # return torch.clamp(x_, 0, 255)
        return x_

    def dc(self, z, b, i):
        z_k = torch.nn.ZeroPad2d(padding=128)(z)

        for k in range(self.K):
            Fz = torch.fft.fft2(z_k) / self.normalization
            absFz = Fz.abs()
            # x_kprime = (torch.fft.ifft2( torch.mul( (torch.sigmoid(self.lam[i])*b + (1-torch.sigmoid(self.lam[i]))*absFz) , Fz / (1e-7+absFz)) )*self.normalization).real
            x_kprime = (
                torch.fft.ifft2(
                    torch.mul(
                        (self.lam[i] * b + (1 - self.lam[i]) * absFz),
                        Fz / (1e-7 + absFz),
                    )
                )
                * self.normalization
            ).real

            indices = torch.logical_or(
                torch.logical_and(x_kprime < 0, self.mask), torch.logical_not(self.mask)
            )  # indices = torch.logical_or(torch.logical_and(torch.logical_or(x_kprime<0, x_kprime>255), self.mask), torch.logical_not(self.mask))

            z_k_new = x_kprime
            z_k_new[indices] = z_k[indices] - self.beta * x_kprime[indices]

            z_k = z_k_new

        return z_k[:, :, self.mask_slice[0], self.mask_slice[1]]
