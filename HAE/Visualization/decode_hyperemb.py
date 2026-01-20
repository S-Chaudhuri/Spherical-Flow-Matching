from notebook_utils.pmath import dist0
import geoopt.manifolds.stereographic.math as gmath
import torch
import os
import sys
from notebook_utils.hae.models.hae_inference import hae
from argparse import Namespace
from PIL import Image
from pytorch_fid import fid_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0,'/home/fvaleau/HAE/Visualization/notebook_utils/hae')
print(torch.cuda.is_available()) 
torch.cuda.set_device(0)

model_path = '/home/fvaleau/HAE/pretrained_models/hae_flowers.pt'
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
# use the first GPU 
opts['device'] = 'cuda:0'
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False

# instantialize model with checkpoints and args
opts = Namespace(**opts)
net = hae(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

emb_name = "fm_samples5"

samples = torch.load(f"/home/fvaleau/riemannian-fm/{emb_name}")
#samples = samples.reshape(1020, 18, 512)     # to comment
decoded_images = []
os.makedirs(f"/home/fvaleau/HAE/FID/{emb_name}", exist_ok=True)

def rescale(target_radius, x):
    r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = x, k = torch.tensor(-1.0)))
    return gmath.mobius_scalar_mul(r = r_change, x = x, k = torch.tensor(-1.0))

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


for z in samples:
    # ensure decoder stability
    r = dist0(z)      # uncomment
    print(r)          # uncomment
    #if r > 6.2126:
        #print("sample is not in the ball")
        #z = rescale(6.2126, z)

    # this is for decoding poincare samples!!!
    with torch.no_grad():
        img, _, _, _, _ = net.forward(
            x = z.unsqueeze(0),
            codes=None,
            batch_size=1,
            input_feature=True,
            input_code=False
        )

    # decoding euclidean samples (W+)
    # with torch.no_grad():
    #     img, _, _, _, _ = net.forward(
    #         x = z.unsqueeze(0),
    #         randomize_noise=False,
    #         input_feature=False,
    #         input_code=True
    #     )

    decoded_images.append(img)



for i in range(len(decoded_images)):
    pil_img = tensor2im(decoded_images[i].squeeze(0))
    pil_img.save(f"/home/fvaleau/HAE/FID/{emb_name}/fm_sample_{i+1}.png")


gen_dir = f"/home/fvaleau/HAE/FID/{emb_name}"
new_test = "/home/fvaleau/HAE/FID/flowers_train"



paths = [new_test, gen_dir]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = fid_score.calculate_fid_given_paths(
    paths,
    batch_size=32,
    device=device,
    dims=2048,
    num_workers=0
)

print(f"FID score: {fid:.4f} for {gen_dir}")