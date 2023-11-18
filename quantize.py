# Quantize U-Net Segmentation Model


import os
from main_training import setup_and_run_train
import torch
from unet import UNet

test_img_dir = "./test_images"
output_dir = "./output"
model_path = "./models/unet_carvana_scale1.0_epoch2.pth"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Initialization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_channels = 1
n_classes = 1
net = UNet(n_channels, n_classes).to("cpu")
print("Loading model...")
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
print(checkpoint)
net.load_state_dict(checkpoint)
m = net.eval()

# Define modules to fuse
## NOTE: I was working with an in-house model with U-Net Architecture 
## built by a previous employee so I spent some time figuring out how
## the model had changed from original creation to loading to my device.

# Here I fuse Conv2d, BatchNorm2d, and ReLU layers into one QuantizedConvReLU2d layer
modules_to_fuse =  [
      # inc.conv.conv
      ['inc.double_conv.0','inc.double_conv.1','inc.double_conv.2'],
      ['inc.double_conv.3','inc.double_conv.4','inc.double_conv.5'],                
        # down1.mpconv.conv
      ['down1.maxpool_conv.1.double_conv.0','down1.maxpool_conv.1.double_conv.1','down1.maxpool_conv.1.double_conv.2'],
      ['down1.maxpool_conv.1.double_conv.3','down1.maxpool_conv.1.double_conv.4','down1.maxpool_conv.1.double_conv.5'],
      # down2.mpconv.conv
      ['down2.maxpool_conv.1.double_conv.0','down2.maxpool_conv.1.double_conv.1','down2.maxpool_conv.1.double_conv.2'],                   
      ['down2.maxpool_conv.1.double_conv.3','down2.maxpool_conv.1.double_conv.4','down2.maxpool_conv.1.double_conv.5'],
      # down3.mpconv.conv
      ['down3.maxpool_conv.1.double_conv.0','down3.maxpool_conv.1.double_conv.1','down3.maxpool_conv.1.double_conv.2'],
      ['down3.maxpool_conv.1.double_conv.3','down3.maxpool_conv.1.double_conv.4','down3.maxpool_conv.1.double_conv.5'],
      # down4.mpconv.conv
      ['down4.maxpool_conv.1.double_conv.0','down4.maxpool_conv.1.double_conv.1','down4.maxpool_conv.1.double_conv.2'],
      ['down4.maxpool_conv.1.double_conv.3','down4.maxpool_conv.1.double_conv.4','down4.maxpool_conv.1.double_conv.5'],
      # up1.conv.conv
      ['up1.conv.double_conv.0','up1.conv.double_conv.1','up1.conv.double_conv.2'],
      ['up1.conv.double_conv.3','up1.conv.double_conv.4','up1.conv.double_conv.5'],
      # up2.conv.conv
      ['up2.conv.double_conv.0','up2.conv.double_conv.1','up2.conv.double_conv.2'],
      ['up2.conv.double_conv.3','up2.conv.double_conv.4','up2.conv.double_conv.5'],
      # up3.conv.conv
      ['up3.conv.double_conv.0','up3.conv.double_conv.1','up3.conv.double_conv.2'],
      ['up3.conv.double_conv.3','up3.conv.double_conv.4','up3.conv.double_conv.5'],
      # up4.conv.conv
      ['up4.conv.double_conv.0','up4.conv.double_conv.1','up4.conv.double_conv.2'],
      ['up4.conv.double_conv.3','up4.conv.double_conv.4','up4.conv.double_conv.5']]


fused_net = torch.quantization.fuse_modules(m, modules_to_fuse)

# set qconfig
fused_net.qconfig = torch.quantization.default_qat_config
fused_net = torch.quantization.prepare_qat(fused_net, inplace=True)

# Finetuning the model
for param in fused_net.parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()

optimizer_ft = torch.optim.SGD(fused_net.parameters(), lr=1e-3, momentum=0.9, ewight_decay=0.1)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)

# Retrain the model through a custom function
og_image_dir = "./original_image"
gt_image_dir = "./ground_truth"
test_image_dir = "./test_dataset"
output_file = "output"
opt = 'Adam'
l = 'Dice'
e = 'Dice'
run = 'first'
val_perc = 0.3
batch_size = 10
ep = 100
lr = 0.00002

model_ft_tuned = setup_and_run_train(fused_net, 1, 1, og_image_dir, gt_image_dir, test_image_dir, model_path, val_perc, batch_size, ep, lr, run, output_file, opt, l, e, run)

model_ft_tuned.cpu()

model_quantized_and_trained = torch.quantization.convert(model_ft_tuned, inplace = False)

# Save quantized model
torch.save(model_quantized_and_trained.state_dict(), output_file+".pth")
