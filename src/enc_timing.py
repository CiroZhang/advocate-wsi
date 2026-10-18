"""Per-patch encoder cost on one A100 (architecture only; weights irrelevant for cost).
UNI = ViT-L/16 (timm vit_large_patch16_224, init_values 1e-5); students = ResNet50 / ResNet18 / MobileNetV3-L (torchvision).
FLOPs via torch.utils.flop_counter (counted per image), throughput in fp16 autocast, batch 256, channels-last for CNNs."""
import time, json, torch, timm, torchvision
import os
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from torch.utils.flop_counter import FlopCounterMode
dev = 'cuda'; torch.backends.cudnn.benchmark = True; out = {}
models = {'UNI_ViT-L16': lambda: timm.create_model('vit_large_patch16_224', init_values=1e-5, num_classes=0, dynamic_img_size=True),
          'ResNet50': lambda: torchvision.models.resnet50(),
          'ResNet18': lambda: torchvision.models.resnet18(),
          'MobileNetV3-L': lambda: torchvision.models.mobilenet_v3_large()}
for name, mk in models.items():
    m = mk().to(dev).eval()
    for res in (224, 112):
        if name.startswith('UNI') and res != 224: continue
        x = torch.randn(1, 3, res, res, device=dev)
        with FlopCounterMode(display=False) as fc, torch.no_grad(): m(x)
        gfl = fc.get_total_flops() / 1e9
        xb = torch.randn(256, 3, res, res, device=dev)
        mm = m.to(memory_format=torch.channels_last) if not name.startswith('UNI') else m
        xb = xb.to(memory_format=torch.channels_last) if not name.startswith('UNI') else xb
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            for _ in range(5): mm(xb)
            torch.cuda.synchronize(); t = time.time(); n = 0
            while time.time() - t < 20: mm(xb); n += 256
            torch.cuda.synchronize(); ips = n / (time.time() - t)
        out[f'{name}@{res}'] = {'gflops_per_patch': round(gfl, 2), 'patches_per_s': round(ips, 1)}
        print(name, res, out[f'{name}@{res}'], flush=True)
json.dump(out, open(ROOT + '/studentB_debate/results/enc_timing.json', 'w'), indent=1)
print('DONE')
