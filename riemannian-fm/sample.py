from manifm.eval_utils import load_model
import torch

#checkpoint = "/home/fvaleau/riemannian-fm/outputs/runs/images/fm/2025.12.05/183800/checkpoints/last.ckpt"
checkpoint = "/home/fvaleau/riemannian-fm/outputs/runs/images/fm/2026.01.12/022417/checkpoints/epoch-394_step-0_loss-0.0000.ckpt"

cfg, model = load_model(checkpoint)
print(type(model.manifold))

model = model.cuda()
model.eval()
with torch.no_grad():
    samples = model.sample(1020, device="cuda")
print(samples.shape)   # torch.Size([10, 512])
print(samples)
torch.save(samples, "fm_samples3")