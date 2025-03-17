import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch import autocast
from contextlib import nullcontext


from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,unetHF_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from src.utils.torch_utils import add_full_name_to_module
import wandb
from scripts.gen_image import gen_image_from_prompt

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from scripts.gen_val_images import gen_images
from scripts.hf15.init_pipe import init_pipe

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. {v} was passed")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )

    parser.add_argument(
        "--quant_act_ops", action="store_true", 
        help="if to quantize  ops activations when ptq==True"
    )

    parser.add_argument(
        "--split_to_16bits", action="store_true", 
        help="replace act split with 16bits acts"
    )

    parser.add_argument(
        "--gen_val_images", type=str,
        default='true',
        help="generate validation images"
    )


    parser.add_argument(
        "--accum_batches", action="store_true", 
        help="accumulate gradients for activation quantization"
    )

    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--symmetric_weight",
        action="store_true",
        help="symetric weight quantization",
    )

    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="symmetric", 
        choices=["linear", "squant", "qdiff","rtn"], 
        help="quantization mode to use"
    )

    parser.add_argument(
        "--naive_weights_quant",type=str, default  = "false",
        help="naive weight quantization"
    )

    parser.add_argument(
        "--rev_order",type=str, default  = "false",
        help="first act quant then weight quant"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="debuf with small dataset"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    if opt.debug:
        opt.outdir = opt.outdir + "-debug"
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    opt.naive_weights_quant = str2bool(opt.naive_weights_quant)
    opt.rev_order = str2bool(opt.rev_order)
    opt.gen_val_images = str2bool(opt.gen_val_images)

    #p_name = "q-diff" if not opt.quant_act_ops else "q-diff-act-ops"
    p_name = "q-diff-hf1.5_verb"
    
    #if opt.ddim_steps != 50:
    #    p_name = p_name + f'ddim_steps-{opt.ddim_steps}'
    if opt.debug:
        p_name =  p_name + "-debug"
    
    run = wandb.init(
            # Set the project where this run will be logged
            project = p_name,
            # Track hyperparameters and run metadata
            config={
                "weight_bit": opt.weight_bit,
                "symmetric_weight": opt.symmetric_weight,
                "act_quant": opt.quant_act,
                "act_quant_ops": opt.quant_act_ops,
                "split_to_16bits": opt.split_to_16bits,
                "accum_batches": opt.accum_batches,
                "act_bit": opt.act_bit,
                "act_quant_mode": opt.quant_mode,
                "naive_weights_quant": opt.naive_weights_quant,
                "rev_order": opt.rev_order,
                "sm_abit": opt.sm_abit,
                "ddim_steps": opt.ddim_steps,
                "resume_w": opt.resume_w,
                "resume": opt.resume,
                "cali_iters_a": opt.cali_iters_a,
                "cali_iters": opt.cali_iters,
                "cali_ckpt": opt.cali_ckpt,
                "cali_data_path": opt.cali_data_path,
                "prompt": opt.prompt,
                "debug": opt.debug, 
                "outpath": outpath, 
            },
    )

    logger = logging.getLogger(__name__)
    logger.info(f"wbit={opt.weight_bit}, sym={opt.symmetric_weight}, act_q={opt.quant_act},abit={opt.act_bit}, sm_abit={opt.sm_abit},{opt.split_to_16bits=}, resume_w={opt.resume_w}, {opt.cali_data_path}")
    

    if opt.resume_w:
        logger.info(f"Resume_w from {opt.resume_w} {opt.cali_ckpt}")
    if opt.resume:
        logger.info(f"Resume from {opt.resume=} {opt.cali_ckpt}")


    #pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE")
    pipe = init_pipe()
    model = pipe.unet

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

  
    assert(opt.cond)
    if opt.ptq:
        if opt.quant_mode == 'qdiff' or opt.quant_mode == 'rtn':
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse',
                         'symmetric':opt.symmetric_weight,'debug':opt.debug}
            aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 
                         'leaf_param':  opt.quant_act, 'debug':opt.debug,'split_to_16bits':opt.split_to_16bits,'act_quant_mode' :opt.quant_mode}
            if opt.naive_weights_quant:
                wq_params['scale_method'] = 'max'

            qnn = QuantModel(
                model=model, weight_quant_params=wq_params, act_quant_params=aq_params,
                act_quant_mode="qdiff", sm_abit=opt.sm_abit,quant_act_ops = opt.quant_act_ops, split=opt.split)
            qnn.cuda()
            qnn.eval()
            # logging.info(qnn)
            add_full_name_to_module(qnn)
        
            if opt.no_grad_ckpt:
                logger.info('Not use gradient checkpointing for transformer blocks')
                qnn.set_grad_ckpt(False)

            if opt.resume:
                cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=opt.cond)
            else:
                logger.info(f"Sampling data from {opt.cali_data_path} at {opt.cali_st} timesteps for calibration")
                sample_data = torch.load(opt.cali_data_path)
                cali_data = get_train_samples(opt, sample_data, opt.ddim_steps)
                del(sample_data)
                if opt.debug:
                    print(f"Calibration data shape debug reduction:")
                    cali_data = [x[:opt.cali_batch_size*2] for x in cali_data]
                    opt.cali_iters = 20 if not opt.accum_batches else 24
                    opt.cali_iters_a = 20 if not opt.accum_batches else 32


                gc.collect()
                logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")

                cali_xs, cali_ts, cali_cs = cali_data

                if not opt.rev_order: # weight quantization first
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda(), cali_cs[:1].cuda())
                    logger.info("Initializing has done!") 
                # Kwargs for weight rounding calibration
            
                    if not opt.naive_weights_quant: # adaptive rounding for weights

                        logger.info("Doing weight calibration")
                        #recon_model(qnn)
                        kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                                iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond,
                                #accum_batches= 4 if opt.accum_batches else 1)
                                accum_batches = 1,rev = opt.rev_order)
                        unetHF_reconstruction(qnn, **kwargs)
                        logger.info(f"finished weight Calibration Saving  checkpoint to {outpath}/wc_ckpt.pth")
                        add_full_name_to_module(qnn)
                        for m in qnn.model.modules():
                            if isinstance(m, AdaRoundQuantizer):
                                m.zero_point = nn.Parameter(m.zero_point)
                                m.delta = nn.Parameter(m.delta)
                        torch.save(qnn.state_dict(), os.path.join(outpath, "wc_ckpt.pth"))
                    else: # naive quant wieghts
                        logger.info("Naive weight quantization allready done in model init")
                        add_full_name_to_module(qnn)
                        for m in qnn.model.modules():
                            if isinstance(m, UniformAffineQuantizer) and 'weight' in m.full_name:
                                m.zero_point = nn.Parameter(m.zero_point)
                                m.delta = nn.Parameter(m.delta)

                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                

                if opt.quant_act:
                    logger.info("UNet model")
                    #logger.info(model.model)                    
                    logger.info(f"Doing activation calibration {opt.quant_mode=}")
                    # Initialize activation quantization parameters
                    qnn.set_quant_state(weight_quant= not opt.rev_order, act_quant=True)
                    with torch.no_grad():
                        act_bs = 8//2
                        inds = np.random.choice(cali_xs.shape[0], act_bs, replace=False)
                        _ = qnn(cali_xs[inds[:act_bs//2]].cuda(), cali_ts[inds[:act_bs//2]].cuda(), cali_cs[inds[:act_bs//2]].cuda())
                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            inds = np.arange(cali_xs.shape[0])
                            np.random.shuffle(inds)
                            qnn.set_running_stat(True, opt.rs_sm_only)
                            for i in trange(int(cali_xs.size(0) / act_bs)):
                                _ = qnn(cali_xs[inds[i * act_bs:(i + 1) * act_bs]].cuda(), 
                                    cali_ts[inds[i * act_bs:(i + 1) * act_bs]].cuda(),
                                    cali_cs[inds[i * act_bs:(i + 1) * act_bs]].cuda())
                            qnn.set_running_stat(False, opt.rs_sm_only)
                        gc.collect()
                    act_bs = opt.cali_batch_size // 2 if not opt.quant_act_ops else opt.cali_batch_size // 4
                    accum_batches =  opt.cali_batch_size // act_bs # 2 if opt.accum_batches else 1
                    kwargs = dict(
                                    cali_data=cali_data, batch_size=opt.cali_batch_size//2, 
                                    iters=opt.cali_iters_a, act_quant=True,opt_mode='mse', 
                                    lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond,
                                    accum_batches= accum_batches,rev = opt.rev_order)
                    if  opt.quant_mode == 'qdiff':
                        unetHF_reconstruction(qnn, **kwargs)
                    elif opt.quant_mode == 'rtn':
                        logger.info("RTN calibration was done in stats collection")
                    else:
                        raise NotImplementedError(f"quant_mode={opt.quant_mode} not implemented")
                    qnn.set_quant_state(weight_quant=not opt.rev_order, act_quant=True)
                
                if opt.rev_order: # weight quantization last
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, True)
                    _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda(), cali_cs[:1].cuda())
                    if not opt.naive_weights_quant: # adaptive rounding for weights
                        logger.info("Doing weight calibration")
                        #recon_model(qnn)
                        weight_bs = opt.cali_batch_size // 4 # if not opt.quant_act_ops else opt.cali_batch_size // 8
                        accum_batches =  opt.cali_batch_size // weight_bs # 2 if opt.accum_batches else 1
                        
                        kwargs = dict(cali_data=cali_data, batch_size=weight_bs, 
                                iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond,
                                #accum_batches= 4 if opt.accum_batches else 1)
                                accum_batches = accum_batches,rev = opt.rev_order)
                        
                        unetHF_reconstruction(qnn, **kwargs)
                        qnn.set_quant_state(True, True)
                        logger.info(f"finished weight Calibration")
                    else: # naive quant wieghts
                        raise NotImplementedError("Naive weight quantization not implemented in rev_order")
                



                logger.info(f"Saving calibrated quantized UNet model to {outpath}/ckpt.pth")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                torch.save(aq_params, os.path.join(outpath, "aq_params.pth"))
                torch.save(wq_params, os.path.join(outpath, "wq_params.pth"))
                torch.save(opt, os.path.join(outpath, "opt.pth"))

            

    n_samples = opt.n_samples 
    n_rows = opt.n_rows if opt.n_rows > 0 else n_samples
    n_iter = opt.n_iter  
    
    qnn.set_quant_state(weight_quant=True, act_quant=opt.quant_act)
    pipe.unet = qnn.model
    pipe.to(device)

    generator = torch.Generator("cuda").manual_seed(42)  # 
    I = pipe(opt.prompt,num_inference_steps=opt.ddim_steps,generator= generator).images[0]
            
    grid_count=0
    I.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
    grid_count += 1
                    #upload image to wandb
    wandb.log({"grid act and weights": [wandb.Image(I)]})
    if opt.gen_val_images :
        I = gen_images(pipe, num_images = 1 if opt.debug else 4,num_inference_steps = opt.ddim_steps, output_image_path = None)
        I.save(os.path.join(outpath, 'grid-val_images.png'))
        wandb.log({"grid val images": [wandb.Image(I)]})




    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
