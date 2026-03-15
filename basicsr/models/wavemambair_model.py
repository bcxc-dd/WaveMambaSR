import torch
from torch.nn import functional as F
from collections import OrderedDict
import os
import cv2
import numpy as np


from basicsr.utils import tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.losses import build_loss

# 引入你的 DWT 模块用于计算频域损失 (如果没有引用，下面我内置了一个简化版)
# from basicsr.archs.gemini_wavemambair import DWT 

@MODEL_REGISTRY.register()
class WaveMambaIRModel(SRModel):
    """
    WaveMambaIR Model for Single Image Super-Resolution.
    继承自 SRModel，但增加了可选的【频域损失 (Wavelet Loss)】。
    """

    def __init__(self, opt):
        super(WaveMambaIRModel, self).__init__(opt)
        
        # 初始化频域损失 (如果在配置文件中开启)
        if self.opt['train'].get('wavelet_opt'):
            self.cri_wave = build_loss(self.opt['train']['wavelet_opt']).to(self.device)
        else:
            self.cri_wave = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # 1. 前向传播
        #self.output,self.ll_feat,self.hf_feat = self.net_g(self.lq)

        logits = self.net_g(self.lq, return_freq=True)
        
        if isinstance(logits, tuple):
            self.output, self.ll_feat, self.hf_feat = logits
        else:
            self.output = logits
            self.ll_feat, self.hf_feat = None, None

        l_total = 0
        loss_dict = OrderedDict()

        # 2. 基础像素损失 (L1 / Charbonnier)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 3. 感知损失 (Perceptual Loss)
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 4. [新增] 频域损失 (Wavelet Loss)
        # 强制 Output 的高频分量和 GT 的高频分量一致
        if self.cri_wave:
            l_wave = self.cri_wave(self.output, self.gt) 
            l_total += l_wave
            loss_dict['l_wave'] = l_wave

        # 5. 反向传播与更新
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        self.log_dict['energy/ll_branch'] = self.ll_feat.detach().abs().mean().item()
        self.log_dict['energy/hf_branch'] = self.hf_feat.detach().abs().mean().item()

        # EMA 更新
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # 保存训练过程中的可视化结果
        save_freq = 10
        rank, _ = get_dist_info()
        
        if current_iter % save_freq == 0 and rank == 0:
            self._save_training_visuals(current_iter)

    #保存逻辑的函数实现
    def _save_training_visuals(self, current_iter):
        vis_dir = os.path.join(self.opt['path']['visualization'], 'train_debug')
        os.makedirs(vis_dir, exist_ok=True)
        
        with torch.no_grad():
            # --- [修复] 保存 LL (低频特征) ---
            if hasattr(self, 'll_feat') and self.ll_feat is not None:
                # 原始数据形状: (Batch, 48, H, W) -> 取第0张: (48, H, W)
                raw_ll = self.ll_feat[0].detach().float().cpu()
                
                # 【关键修改】不能直接 tensor2img，必须手动降维
                # 对 48 个通道取平均，变成 (H, W) 的单通道图
                ll_map = raw_ll.mean(dim=0).numpy()
                
                # 归一化到 0-255
                min_v, max_v = ll_map.min(), ll_map.max()
                if max_v - min_v > 1e-8:
                    ll_map = (ll_map - min_v) / (max_v - min_v)
                else:
                    ll_map = np.zeros_like(ll_map)
                
                ll_img = (ll_map * 255).astype(np.uint8)
                
                # 为了看清结构，用伪彩色 (Viridis 配色适合看特征)
                ll_color = cv2.applyColorMap(ll_img, cv2.COLORMAP_VIRIDIS)
                
                save_path = os.path.join(vis_dir, f'iter_{current_iter}_LL.png')
                cv2.imwrite(save_path, ll_color)

            # --- 保存 HF (高频特征) ---
            if hasattr(self, 'hf_feat') and self.hf_feat is not None:
                # HF 也是多通道 (144)，处理逻辑同上
                # 取绝对值是因为高频残差有正有负，我们只关心“强度”
                hf_map = self.hf_feat[0].detach().abs().mean(dim=0).cpu().numpy()
                
                min_v, max_v = hf_map.min(), hf_map.max()
                if max_v - min_v > 1e-8:
                    hf_map = (hf_map - min_v) / (max_v - min_v)
                else:
                    hf_map = np.zeros_like(hf_map)
                
                hf_img = (hf_map * 255).astype(np.uint8)
                # Jet 配色适合看热力图 (红强蓝弱)
                hf_color = cv2.applyColorMap(hf_img, cv2.COLORMAP_JET)
                
                save_path = os.path.join(vis_dir, f'iter_{current_iter}_HF.png')
                cv2.imwrite(save_path, hf_color)

            # --- 保存 SR 结果 (这是 RGB图，可以直接用 tensor2img) ---
            if hasattr(self, 'output') and self.output is not None:
                sr_img = tensor2img(self.output[0].detach())
                save_path = os.path.join(vis_dir, f'iter_{current_iter}_SR.png')
                cv2.imwrite(save_path, sr_img)

    def get_current_visuals(self):
        """将中间特征转换为可视化图像"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        
        # 可视化低频分支 (LL)
        # LL 通常是图像的缩略版，直接记录
        if hasattr(self, 'll_feat'):
            out_dict['vis_ll'] = self.ll_feat.detach().cpu()
            
        # 可视化高频分支 (HF)
        # HF 是残差形式，均值接近0。我们通过取绝对值并放大来观察纹理分布
        if hasattr(self, 'hf_feat'):
            # 对多个高频通道取平均，变成单通道热力图
            hf_vis = self.hf_feat.detach().cpu().abs().mean(dim=1, keepdim=True)
            # 归一化到 0-1 方便查看
            hf_vis = (hf_vis - hf_vis.min()) / (hf_vis.max() - hf_vis.min() + 1e-8)
            out_dict['vis_hf'] = hf_vis

        return out_dict
    
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]            