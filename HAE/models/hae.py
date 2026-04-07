
from audioop import bias
import matplotlib
matplotlib.use('Agg')
import math
# import geoopt.manifolds.stereographic.math as gmath


import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
# from models.hyper_nets import MobiusLinear, HyperbolicMLR
from configs.paths_config import model_paths


def vector_rms_norm(z, zero_mean=False, eps=1e-6, curvature=1.0):
    """
    L2 normalization to project vectors to sphere with given curvature.
    Curvature k > 0: sphere with radius r = 1/sqrt(k)
    From reference, adapted for variable curvature.
    """
    assert z.ndim >= 2
    dim = tuple(range(1, z.ndim))
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    # Compute L2 norm
    norm = torch.sqrt(z.square().sum(dim=dim, keepdim=True) + eps)
    # Normalize to unit sphere first, then scale to desired radius
    radius = 1.0 / torch.sqrt(torch.tensor(curvature, dtype=z.dtype, device=z.device))
    return z * (radius / norm)


def vector_compute_magnitude(x):
    """Compute L2 norm (from reference)"""
    assert x.ndim >= 2
    reduce_dims = tuple(range(1, x.ndim))
    mag = x.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    return mag


def vector_compute_angle(x, y):
    """Compute angle between two vectors on sphere (from reference)"""
    assert x.ndim >= 2
    assert x.shape == y.shape
    reduce_dims = tuple(range(1, x.ndim))
    dot = (x * y).sum(dim=reduce_dims, keepdim=True)
    mag = vector_compute_magnitude(x) * vector_compute_magnitude(y)
    mag = torch.clamp(mag, min=1e-6)
    cos_sim = torch.clamp(dot / mag, min=-1.0, max=1.0)
    angle = torch.acos(cos_sim) / math.pi * 180.0
    return angle


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class EqualLinear_encoder_low(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_encoder_low, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*3, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, in_dim)
        self.fc5 = nn.Linear(in_dim, out_dim*2)
        self.fc6 = nn.Linear(out_dim*2, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class EqualLinear_encoder_mid(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_encoder_mid, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*4, in_dim*3)
        self.fc2 = nn.Linear(in_dim*3, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, in_dim)
        self.fc5 = nn.Linear(in_dim, out_dim*2)
        self.fc6 = nn.Linear(out_dim*2, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class EqualLinear_encoder_high(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_encoder_high, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*11, in_dim*8)
        self.fc2 = nn.Linear(in_dim*8, in_dim*6)
        self.fc3 = nn.Linear(in_dim*6, in_dim*4)
        self.fc4 = nn.Linear(in_dim*4, in_dim*2)
        self.fc5 = nn.Linear(in_dim*2, in_dim)
        self.fc6 = nn.Linear(in_dim, in_dim)
        self.fc7 = nn.Linear(in_dim, out_dim*2)
        self.fc8 = nn.Linear(out_dim*2, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        out = self.nonlinearity(out)
        out = self.fc7(out)
        out = self.nonlinearity(out)
        out = self.fc8(out)
        return out

class EqualLinear_decoder_low(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_decoder_low, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, out_dim*2)
        self.fc5 = nn.Linear(out_dim*2, out_dim*2)
        self.fc6 = nn.Linear(out_dim*2, out_dim*3)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class EqualLinear_decoder_mid(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_decoder_mid, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, out_dim*2)
        self.fc5 = nn.Linear(out_dim*2, out_dim*3)
        self.fc6 = nn.Linear(out_dim*3, out_dim*4)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class EqualLinear_decoder_high(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_decoder_high, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, out_dim*2)
        self.fc5 = nn.Linear(out_dim*2, out_dim*4)
        self.fc6 = nn.Linear(out_dim*4, out_dim*6)
        self.fc7 = nn.Linear(out_dim*6, out_dim*8)
        self.fc8 = nn.Linear(out_dim*8, out_dim*11)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        out = self.nonlinearity(out)
        out = self.fc7(out)
        out = self.nonlinearity(out)
        out = self.fc8(out)
        return out


class MLP_encoder(nn.Module):
    def __init__(self, dim):
        super(MLP_encoder, self).__init__()
        self.encoder_low=EqualLinear_encoder_low(512, dim)
        self.encoder_mid=EqualLinear_encoder_mid(512, dim)
        self.encoder_high=EqualLinear_encoder_high(512, dim*2)
        
    def forward(self, dw):
        x0=self.encoder_low(dw[:, :3])
        x1=self.encoder_mid(dw[:, 3:7])
        x2=self.encoder_high(dw[:, 7:])
        output_dw = torch.cat((x0, x1, x2), dim=1)
        return output_dw

class MLP_decoder(nn.Module):
    def __init__(self, dim):
        super(MLP_decoder, self).__init__()
        self.dim = dim
        self.decoder_low=EqualLinear_decoder_low(dim, 512)
        self.decoder_mid=EqualLinear_decoder_mid(dim, 512)
        self.decoder_high=EqualLinear_decoder_high(dim*2, 512)    
  
    def forward(self, dw):
        shape = dw[:, :self.dim].shape
        x0=self.decoder_low(dw[:, :self.dim])
        x1=self.decoder_mid(dw[:, self.dim:self.dim*2])
        x2=self.decoder_high(dw[:, self.dim*2:])
        dw0 = x0.reshape((shape[0], 3, 512))
        dw1 = x1.reshape((shape[0], 4, 512))
        dw2 = x2.reshape((shape[0], 11, 512))
        output_dw = torch.cat((dw0, dw1, dw2), dim=1)
        return output_dw
        


class hae(nn.Module):

    def __init__(self, opts):
        super(hae, self).__init__()
        self.set_opts(opts)
        
        # Spherical curvature parameter: k > 0 gives sphere of radius 1/sqrt(k)
        # k = 1.0 is unit sphere, higher k = smaller radius, lower k = larger radius
        self.curvature = torch.tensor(self.opts.spherical_curvature, dtype=torch.float32)
        
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.feature_shape = self.opts.feature_size
        if self.opts.dataset_type == 'flowers_encode'or self.opts.dataset_type == 'flowers_encode_eva':
            self.num_classes = 102 #animal_faces 151, flowers 102
        elif self.opts.dataset_type == 'animalfaces_encode' or self.opts.dataset_type == 'animalfaces_encode_eva':
            self.num_classes = 151
        elif self.opts.dataset_type == 'vggfaces_encode' or self.opts.dataset_type == 'vggfaces_encode_eva':
            self.num_classes = 1802
        else:
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        self.mlp_encoder = MLP_encoder(128)
        self.mlp_decoder = MLP_decoder(128)
        
    
        self.spherical_linear = nn.Linear(self.feature_shape, self.feature_shape)
        
        # Using learned class prototypes with cosine similarity approach
        self.class_prototypes = nn.Parameter(torch.randn(self.num_classes, self.feature_shape))
        nn.init.kaiming_uniform_(self.class_prototypes, a=math.sqrt(5))
        
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading HAE from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('module.','') 
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            # Load converted layers (hyperbolic_linear -> spherical_linear, mlr -> class_prototypes)
            self.spherical_linear.load_state_dict(get_keys(ckpt, 'spherical_linear'), strict=False)
            try:
                self.class_prototypes.data = get_keys(ckpt, 'class_prototypes')['data']
            except:
                pass  # May not exist if converting from hyperbolic checkpoint
            self.mlp_encoder.load_state_dict(get_keys(ckpt, 'mlp_encoder'), strict=False)
            self.mlp_decoder.load_state_dict(get_keys(ckpt, 'mlp_decoder'), strict=False)
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading pSp from checkpoint: {}'.format(self.opts.psp_checkpoint_path))
            ckpt = torch.load(self.opts.psp_checkpoint_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('.module','') 
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)
            self.__load_latent_avg(ckpt)

    def forward(self, x, y = None, batch_size=4, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, input_feature = False):
        if not input_feature:
            if input_code:
                codes = x
            else:
                codes = self.encoder(x)
                # normalize with respect to the center of an average face
                if self.opts.start_from_latent_avg:
                    if self.opts.learn_in_w:
                        codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                    else:
                        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

            ocodes = codes
            feature = self.mlp_encoder(ocodes)
            feature_reshape = torch.flatten(feature, start_dim=1)
            # Spherical projection: linear -> sphere projection with curvature
            feature_dist = self.spherical_linear(feature_reshape)
            feature_dist = vector_rms_norm(feature_dist, curvature=self.curvature.item())

        else:
            feature_dist = x
            #codes = self.encoder(y)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        
        # Compute logits using cosine similarity with class prototypes
        # feature_dist: [B, D], class_prototypes: [num_classes, D]
        normalized_prototypes = vector_rms_norm(self.class_prototypes, curvature=self.curvature.item())
        # Both vectors are on sphere, compute cosine similarity (dot product)
        logits = torch.mm(feature_dist, normalized_prototypes.t())  # [B, num_classes]
        logits = F.log_softmax(logits, dim=-1)
        
        # For model output, feature_euc is just the spherical feature (not decoding to euclidean)
        feature_euc = feature_dist
        feature_euc = self.mlp_decoder(feature_euc)
        codes = feature_euc
        #codes = torch.cat((feature_euc, codes[:, 6:]), dim=1)


        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent, logits, feature_dist, ocodes, feature_euc
        else:
            return images, logits, feature_dist, ocodes, feature_euc

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
