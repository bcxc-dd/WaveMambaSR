import os
import torch
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation
from models.team12_DWMamba import DWMamba

def main():
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Testing Device: {device}")

    
    print("[*] Initializing DWMamba...")
    model = DWMamba(
        upscale=4,
        in_chans=3,
        img_range=1.0,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[2, 2, 2, 2],
        num_heads=[4, 4, 4, 4], 
        window_size=16, 
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=2.0,
        upsampler='pixelshuffledirect',
        resi_connection='1conv' 
    ).eval().to(device)

    model_path = os.path.join('model_zoo', 'team12_DWMamba.pth')
    if os.path.exists(model_path):
        stat_dict = torch.load(model_path, map_location=device)
        weight_dict = stat_dict.get('params', stat_dict.get('params_ema', stat_dict))
        model.load_state_dict(weight_dict, strict=True)
        print(f"[*] Successfully loaded weights from {model_path}")
    else:
        print(f"[!] Warning: Weights not found at {model_path}. Using random initialization.")

    
    for param in model.parameters():
        param.requires_grad = False

    
    input_shape = (1, 3, 256, 256) 
    dummy_input = torch.rand(input_shape).to(device)
    print(f"\n[*] Input Tensor Shape: {input_shape}")


   
    num_parameters = sum(p.numel() for p in model.parameters()) / 10**6
    print(f"[+] Params:      {num_parameters:.4f} M")

    
    flops = FlopCountAnalysis(model, dummy_input).total() / 10**9
    print(f"[+] FLOPs:       {flops:.4f} G")

    
    try:
        activations, num_conv = get_model_activation(model, input_shape[1:])
        print(f"[+] Activations: {activations / 10**6:.4f} M")
        print(f"[+] Conv2d Num:  {num_conv}")
    except Exception as e:
        print(f"[!] Activations calc failed: {e}")

    
    _ = model(dummy_input)
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"[+] Max Memory:  {max_mem:.2f} MB")

    with open('model_efficiency_report.txt', 'w') as f:
        f.write("=== DWMamba Efficiency Report ===\n")
        f.write(f"Input Shape:  {input_shape}\n")
        f.write(f"Params:       {num_parameters:.4f} M\n")
        f.write(f"FLOPs:        {flops:.4f} G\n")
        f.write(f"Max Memory:   {max_mem:.2f} MB\n")
    print("\n[*] All done! Report saved to 'model_efficiency_report.txt'")

if __name__ == "__main__":
    main()