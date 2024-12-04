import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.VRDM.base.modules.diffusionmodules.openaimodel import UNetModel
# from model.VRDM.base.modules.encoders.modules import SpatialRescaler
from ResNet50_v1 import model
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
    device = torch.device("cuda")

def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).squeeze(2).to('cpu', torch.uint8).numpy()
    # image = image.permute(1, 2, 0).squeeze(2).cpu().numpy().astype(np.uint8)
    im = Image.fromarray(image)
    im.save(os.path.join(save_path, file_name))

class VRDM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type#l1 loss
        self.objective = model_params.objective#grad

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)

            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # index1,index = model(x)
        # index1,index = torch.max(index, dim=1)
        # # print("=====",index1,index)
        # context = Image.open(f'/home/back_door/data7000/template/{index.item()}.png')
        # context = transforms.ToTensor()(context)

        # plt.figure()
        # plt.imshow(x.squeeze(1).squeeze(0).cpu(),cmap='gray')
        # plt.savefig('1.png')
        # plt.figure()
        # plt.imshow(context.squeeze(1).squeeze(0).cpu(),cmap='gray')
        # plt.savefig('2.png')


        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)

        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t,noise=None):


        # noise = default(noise, lambda: torch.randn_like(x0))
        noise1 = torch.randn_like(x0)
        noise2 = torch.randn_like(x0)
        noise_Rayleigh = torch.sqrt(torch.pow(noise1,2) + torch.pow(noise2,2))
        # noise_Rayleigh = torch.tensor(np.random.rayleigh(scale=1.0,size=x0.size()),dtype=torch.float32).to(device)
        # noise_Rayleigh = torch.clamp(noise_Rayleigh, 0, 1)

        # print("=======",self.m_t)

        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            # objective = (1.0 - m_t+0.0010)*x0
            # objective = m_t * (y - x0) + sigma_t * noise_Rayleigh
            # objective = m_t * (y - x0) + sigma_t * noise1
            objective = m_t * (y - x0) 
            # objective = m_t * (y - x0) + sigma_t * noise1
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()
        
        '''
        version 1
        '''
        
        # objective_x = (1. - m_t) * x0 + m_t * y + sigma_t * noise_Rayleigh
        # objective_y = (1. - m_t) * x0 + m_t * y + sigma_t * noise1

        # objective_z = torch.sqrt(torch.pow(objective_x,2)+torch.pow(objective_y,2))

        '''
        version 2
        '''
        # objective_z = (1.0-m_t)*torch.sqrt(torch.tensor(torch.pi/2.0)) + (1.0 - m_t) * x0 + m_t * y + sigma_t * noise_Rayleigh
        objective_z = (1.0 - m_t) * x0 + m_t * y + sigma_t * noise_Rayleigh
        # objective_z = (1.0 - m_t) * x0 + m_t * y 

        return (
            objective_z,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        # m_t = extract(self.m_t.to(device), t.to(device), x_t.shape)
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        # print("===========",self.num_timesteps)
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):

            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)

            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False,T=0):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:

            '''
            t:   tensor([991], device='cuda:0')
            n_t: tensor([999], device='cuda:0')
            '''
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            
            self.m_t = self.m_t.to(device)

            self.variance_t = self.variance_t.to(device)


            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise1 = torch.randn_like(x_t)
            
            noise2 = torch.randn_like(x_t)
            noise = torch.sqrt(torch.pow(noise1,2)+torch.pow(noise2,2))

            noise_Rayleigh_nt = torch.tensor(np.random.rayleigh(scale=1.0,size=x_t.size()),dtype=torch.float32).to(device)
            noise_Rayleigh_t = torch.tensor(np.random.rayleigh(scale=1.0,size=x_t.size()),dtype=torch.float32).to(device)
            # noise_Rayleigh = torch.clamp(noise_Rayleigh, 0, 1)
            # noise_Rayleigh = torch.sqrt(torch.pow(noise1,2) + torch.pow(noise2,2))

            '''
            vrdm model sampling
            '''
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + (torch.sqrt(var_nt) - sigma2_t )/ (torch.sqrt(var_t)) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)
            
            '''
            vrdm model sampling1
            '''
            # x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y 
    

            return x_tminus_mean, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised,T=len(self.steps))
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            # print("self.steps",len(self.steps))
            img = y
            # import matplotlib.pyplot as plt

            # plt.imshow(y.squeeze(1).squeeze(0).cpu(),cmap='gray')
            # plt.savefig("0.png")
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised,T=len(self.steps))
            #     save_single_image(img.squeeze(0),f'/home/back_door/bbdm/out/',f'{i}.png')
            
            # import matplotlib.pyplot as plt

            # plt.imshow(img.squeeze(1).squeeze(0).cpu(),cmap='gray')
            # plt.savefig("1.png")
            # print(1/0)
            return img

    @torch.no_grad()
    def sample(self, y,step = 2000, context=None, clip_denoised=True, sample_mid_step=False):
        # print("====",self.num_timesteps)
        # self.sample_step = step 
        # self.num_timesteps = step 
        # print("=====",step)
        # self.steps = step
        

        self.register_schedule()
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)