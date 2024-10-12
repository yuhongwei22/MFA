'''
Using MFA to Attack Inpainting Task
'''
import argparse, os, sys, glob
from collections import OrderedDict
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch
import math
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader
from dataset import InpaintingDataset
from ldm.util import default
import time

def attack_img(image, mask, model, device,step_size=0.003, epsilon=8/255, attack_steps=70, sample_type="VT", target_step=800,attack="MFA"):
    '''
    step_size: the step size when optimizing adversarial examples
    epsilon: the l∞ limits when optimizing adversarial examples
    attack steps: the iterations of optimizings
    sample_type: the sample type of choosing t
    target_step: the specified step of attacking
    attack: attack Methods
    '''

    model.eval()
    image.to(device)
    mask.to(device)

    #init noise
    random_init = torch.rand_like(image)*0.003
    adv_img = Variable(torch.clamp(image.data+random_init.data,-1.0,1.0), requires_grad=True)

    for t in range(attack_steps):
        model.zero_grad()
        adv_img_ = torch.clamp((adv_img+1.0)/2.0,
                                    min=0.0, max=1.0)
        mask_ = torch.clamp((mask+1.0)/2.0,
                                min=0.0, max=1.0)
        
        adv_masked_image = (1-mask_)* adv_img_
        adv_masked_image = ((adv_masked_image*2.0)-1.0)

        c = model.cond_stage_model.encode(adv_masked_image)
        cc = torch.nn.functional.interpolate(mask,
                                            size=c.shape[-2:])
        
        encoder_posterior = model.encode_first_stage(image)
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        adv_c = torch.cat((c,cc), dim=1)

        if attack == "EmbeddingAttack":
            '''Maximize the gap between adv_img and clean_img through the Encoder'''

            # Calculate the condition value for Clean
            clean_img = torch.clamp((image+1.0)/2.0,
                                    min=0.0, max=1.0)
            clean_masked_image = (1-mask_) * clean_img
            clean_masked_image = ((clean_masked_image*2.0)-1.0)
            clean_con = model.cond_stage_model.encode(clean_masked_image)
            clean_cc = torch.nn.functional.interpolate(mask,
                                            size=c.shape[-2:])
            
            clean_con = torch.cat((clean_con,clean_cc), dim=1)


            loss_mse = torch.nn.MSELoss()
            loss = loss_mse(adv_c, clean_con)
        else:
            '''Choosing the sample type'''
            if sample_type == "random":
                training_loss, loss_dict, predicted_noise = model(z, adv_c)
            elif sample_type == "target":
                training_loss, loss_dict, predicted_noise = model(z, adv_c,target_step)

            if attack == "MFA":
                '''Maximize the mean of the estimated noise 
                When the mean reduce it turns to Green
                When the mean increase it turns to Purple
                '''
                # Green
                loss = -predicted_noise.mean()
                #Purple
                # loss = predicted_noise.mean() 

            elif attack == "AdvDM":
                loss = training_loss

        loss.backward()

        with torch.no_grad():
            grad_info = step_size * adv_img.grad.data.sign()
            adv_img = adv_img.data + grad_info
            eta = torch.clamp(adv_img.data - image.data, -epsilon, epsilon)
            adv_img = image.data + eta
        
        adv_img = Variable(torch.clamp(adv_img, -1, 1), requires_grad=True)
    
    return adv_img, loss


def main(args, atimg_path, atimg_save,sample_type="random",target_step=800,attack="MFA", step_size=2/255, attack_steps=70, epsilon=8/255):
    '''
    The main function for attacking.
    atimg_path: The path for saving adv_img
    atimg_save: The path for saving the result of adv_img
    step_size: the step size when optimizing adversarial examples
    epsilon: the l∞ limits when optimizing adversarial examples
    attack steps: the iterations of optimizings
    '''
    
    BATCH_SIZE = 1

    #data prepare
    dataset = InpaintingDataset(args.indir, data_len=2000)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    # CUDA setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #Load model
    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    model = model.to(device)
    sampler = DDIMSampler(model)

    # Starting attacking
    with tqdm(total=len(dataloader)) as t:
        for batch in dataloader:
            # Prepare the data
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            image = image*2.0-1.0
            mask = mask*2.0-1.0
            image_path = batch['image_path']
            image_name = os.path.split(image_path[0])[-1]

            # save path
            save_path = atimg_save + image_name
            save_path_ori = atimg_path + image_name

            adv_img, loss = attack_img(image,mask,model,device,
                                       sample_type=sample_type,target_step=target_step,attack=attack,
                                       attack_steps=attack_steps,epsilon=epsilon,step_size=step_size)
            # print(image_name)
            
            t.update(len(image))

            # Generate the result of adv_img
            with torch.no_grad():
                with model.ema_scope():
                    mask_ = torch.clamp((mask+1.0)/2.0,
                                        min=0.0, max=1.0)
                    
                    at_img_ = torch.clamp((adv_img+1.0)/2.0, min=0.0,max=1.0)
                    at_masked_image = (1-mask_)*at_img_
                    at_masked_image = ((at_masked_image*2.0)-1.0)

                    c = model.cond_stage_model.encode(at_masked_image)
                    cc = torch.nn.functional.interpolate(mask,
                                                     size=c.shape[-2:])
                    c = torch.cat((c, cc), dim=1)

                    shape = (c.shape[1]-1,)+c.shape[2:]
                    samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)
                    
                    # save result
                    inpainted = (1-mask_)*at_img_ + mask_*predicted_image
                    inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                    Image.fromarray(inpainted.astype(np.uint8)).save(save_path)
                    inpainted = at_img_.cpu().numpy().transpose(0,2,3,1)[0]*255
                    Image.fromarray(inpainted.astype(np.uint8)).save(save_path_ori)



    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--target_step",
        type=int,
        default=800,
        help="The specified attacking step when optimizing adversarial examples. For LDM T=(0,1000)",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="random",
        help="The sample type of optimizing adversarial examples.[random, target]"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="MFA",
        help="Attacking methods,[AdvDM, MFA, Embedding attack, clean]",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default="2/255",
        help="The step size when optimizing adversarial examples",
    )
    parser.add_argument(
        "--attacksteps",
        type=int,
        default="70",
        help="The iterations of optimizings",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default="8/255",
        help="The l∞ limits when optimizing adversarial examples",
    )

    opt = parser.parse_args()
    # Create the savepath of adv_img and the result of adv_img
    adv_img_path = opt.outdir +'/atimg/'
    adv_img_save = opt.outdir +'/save/'
    os.makedirs(adv_img_path, exist_ok=True)
    os.makedirs(adv_img_save, exist_ok=True)
    sample_type = opt.sample_type
    target_step = opt.target_step
    attack_type = opt.attack
    step_size = opt.step_size
    attack_steps = opt.attacksteps
    epsilon = opt.epsilon

    main(opt, adv_img_path, adv_img_save,
         sample_type=sample_type,target_step=target_step,attack=attack_type,
         step_size=step_size, attack_steps=attack_steps, epsilon=epsilon)