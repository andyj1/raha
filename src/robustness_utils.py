import io
import torch
from PIL import Image
import torch.nn.functional as F

def jpeg_compress_batch(x, quality=50):
    """
    x: torch.Tensor [B,3,H,W] in [0,1] (float32) on CPU or GPU
    returns: torch.Tensor [B,3,H,W] in [0,1] float32 (same device as input)
    """
    device = x.device
    x_cpu = (x.detach().clamp(0,1) * 255.0).to(torch.uint8).cpu()  # [B,3,H,W]

    out = []
    for img in x_cpu:
        pil = Image.fromarray(img.permute(1,2,0).numpy())  # HWC
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        pil_j = Image.open(buf).convert("RGB")
        t = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(pil_j.tobytes())).numpy())
        t = t.view(pil_j.size[1], pil_j.size[0], 3).permute(2,0,1).contiguous()
        out.append(t)

    y = torch.stack(out, dim=0).float() / 255.0
    return y.to(device)

def add_gaussian_noise_image(x, sigma=0.02):
    """
    x: [B,3,H,W] in [0,1]
    sigma: std in [0,1] scale
    """
    noise = sigma * torch.randn_like(x)
    return (x + noise).clamp(0, 1)

def add_awgn_text_embed(z, sigma=0.05):
    """
    z: [B,D] unit-normalized
    sigma: std of Gaussian noise
    """
    z_noisy = z + sigma * torch.randn_like(z)
    return F.normalize(z_noisy, dim=1)

def quantize_unit_embed(z, bits=8):
    """
    z: [B,D] unit-normalized, roughly in [-1,1]
    bits: 8, 6, 4
    """
    z = z.clamp(-1, 1)
    qmax = (2 ** (bits - 1)) - 1  # e.g., 127 for 8-bit signed
    z_q = torch.round(z * qmax) / qmax
    return F.normalize(z_q, dim=1)
