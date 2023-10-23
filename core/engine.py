import os
import torch
import torch.nn as nn
import torch.quantization as qt
import shutil   # 파일 및 디렉터리 작업을 수행하기 위한 함수를 제공하는 모듈
import json
import matplotlib.pyplot as plt

from time import time
from pytz import timezone
from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from data.dataset import get_dataloader
from model import get_model
from utils.metric import AverageMeter
from utils import innopeak_loss
from torchmetrics.functional.image import peak_signal_noise_ratio, \
        structural_similarity_index_measure
import warnings
ignored_warnings = [
    "Please use quant_min and quant_max to specify the range for observers",
    "_aminmax is deprecated as of PyTorch 1.11"
]
for warning_message in ignored_warnings:
    warnings.filterwarnings("ignore", category=UserWarning, message=warning_message)

class Trainer:
    def __init__(self, args):
        # backup args
        self.args = args
        self.start_time = None
        # prepare data pipeline
        self.train_loader, self.valid_loader = get_dataloader(
            root=args.data_path,
            check=args.file_check,
            preload=args.preload,
            normalization=args.normalization,
            batch_size=args.batch_size,
            workers=args.workers,
            hr_size=args.hr_size,
            lr_size=args.lr_size
        )
        # define model
        self.qat = args.qat
        self.model = get_model(args)
        if not self.qat:
            self.model_name = args.model
        else:
            self.model_name = args.model + '_QAT'
        
        if args.load is not None:
            try:
                # load_state_dict() : 모델의 가중치(weight) 및 편향(bias) 등을 로드하는 PyTorch 함수
                self.model.load_state_dict(torch.load(args.load))
            except:
                print(f"Failed to load weights")
        
        self.criterion = self.get_criterion(args)
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.device = torch.device(args.device)
        self.epochs = args.epochs
        
        self.exp_dir = None
        self.save_dir = None
        self.save_interval = args.save_interval
        
        # metrics buffer area
        self.train_loss_data = []
        self.valid_psnr_data = []
        self.valid_ssim_data = []

    def get_criterion(self, args):
        if args.loss == 'l1':
            return nn.L1Loss()      # 평균 절대 오차
        elif args.loss == 'l2':
            return nn.MSELoss()     # 평균 제곱 오차
        elif args.loss == 'inno_loss':
            return innopeak_loss.InnoPeak_loss()

    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            optimizer = Adam(self.model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, args.step, args.gamma)
            # scheduler = CosineAnnealingWarmUpRestarts(
            #     optimizer, 20, 1, args.lr, 0, args.gamma
            # )
            return optimizer, scheduler
        
        elif args.optimizer == 'sgd':
            optimizer = SGD(self.model.parameters(), momentum=args.momentum, lr=args.lr)
            scheduler = StepLR(optimizer, args.step, args.gamma)    # step 마다 lr * gamma를 한다.
            return optimizer, scheduler
    
    def train(self):
        self._prepare()
        self.start_time = time()
        for e in range(1, self.epochs+1):
            print(f"# Train Super Resolution Model [{e}/{self.epochs}]")
            self._train()
            self._valid()
            # print("\033[F\033[J",end="")
            if e % self.save_interval == 0:
                self._save_model(e)
                
        # after train finished
        self._finish()
    
    def _prepare(self):
        self.model.to(self.device)
        if self.qat == True:
            self.model.qconfig = qt.get_default_qat_qconfig('fbgemm')
            self.model.qscheme = torch.per_channel_symmetric
            self.model.quant = qt.QuantStub()
            self.model.dequant = qt.DeQuantStub()
            self.model = qt.prepare_qat(self.model)

        if not os.path.exists("./run"):
            os.makedirs("./run")
        
        exp_name = self.args.model + "-" + datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H-%M")
        exp_dir = os.path.join("./run", exp_name)
        
        # remove duplicate dir
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)      # exp_dir 경로에 해당하는 디렉터리와 모든 하위 파일들을 삭제
        
        os.makedirs(exp_dir)
        self.exp_dir = exp_dir
        os.makedirs(os.path.join(exp_dir, "weights"))
        self.save_dir = os.path.join(exp_dir, "weights")
        
        args_dict = vars(self.args)     # vars() : 인자로 전달된 객체의 속성을 Dictionary 형태로 반환
        with open(os.path.join(exp_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)
        
    def _train(self):
        self.model.train()  # layer들을 training mode로 바꿔준다.
        train_bar = tqdm(self.train_loader, ncols=120)
        train_loss_meter = AverageMeter()   # Train 과정 중 평균 loss, accuracy 등을 계산하는데 사용되는 클래스
        for lr_tensor, hr_tensor in train_bar:
            # Low & High resolution Tensor 이미지를 CPU or GPU로 위치를 옮긴다.
            lr_tensor, hr_tensor = lr_tensor.to(self.device), hr_tensor.to(self.device)
            self.optimizer.zero_grad()
            
            # forward
            sr_tensor = self.model(lr_tensor)

            # calculate loss
            loss = self.criterion(sr_tensor, hr_tensor)
            
            # backpropagation
            loss.backward()                     # 역전파 학습
            self.optimizer.step()               # 기울기 업데이트
            train_loss_meter.update(loss)       
            train_bar.set_description(
                f"# TRAIN : loss(avg)={train_loss_meter.avg:.5f}"
            )
        # print("\033[F\033[J",end="")
        self.train_loss_data.append(train_loss_meter.extract())
        del train_loss_meter
        self.scheduler.step()
        
    def _valid(self):
        self.model.eval()   # layer들을 evaluation mode로 바꿔준다.
        valid_bar = tqdm(self.valid_loader, ncols=120)
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        with torch.no_grad():
            for lr_tensor, hr_tensor in valid_bar:
                lr_tensor, hr_tensor = lr_tensor.to(self.device), hr_tensor.to(self.device)
                sr_tensor = self.model(lr_tensor)
                
                psnr_value = peak_signal_noise_ratio(sr_tensor, hr_tensor)
                ssim_value = structural_similarity_index_measure(sr_tensor, hr_tensor)
                
                psnr_meter.update(psnr_value)
                ssim_meter.update(ssim_value)
                
                valid_bar.set_description(
                    f"# VALID : PSNR={psnr_meter.avg:.3f}, SSIM={ssim_meter.avg:.5f}"
                )
        self.valid_psnr_data.append(psnr_meter.extract())
        self.valid_ssim_data.append(ssim_meter.extract())
        # print("\033[F\033[J",end="")
        del psnr_meter
        del ssim_meter        
    
    def _save_model(self, e:int):
        model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_{e}.pth")
        if os.path.exists(model_weight_path):
            os.remove(model_weight_path)
        # Dictionary 형태로 저장 -> 나중에 로드할 때는 torch.load(), load_state_dict()를 사용한다.
        if self.qat == True:
            knot_model = self.model.to(self.device).eval()
            knot_model = qt.convert(knot_model)
            torch.jit.save(torch.jit.script(knot_model), model_weight_path)
            del knot_model
        else:
            torch.save(self.model.state_dict(), model_weight_path)      
        
    def _finish(self):
        # save final model weights
        model_weight_path = os.path.join(self.save_dir, f"{self.model_name}_final.pth")
        if os.path.exists(model_weight_path):
            os.remove(model_weight_path)
        if self.qat == True:
            self.model = qt.convert(self.model)
            torch.jit.save(torch.jit.script(self.model), model_weight_path)
        else:
            torch.save(self.model.state_dict(), model_weight_path) 
        
        # find min/max psnr/ssim value
        psnr_min = 100.0
        psnr_max = 0.0
        ssim_min = 100.0
        ssim_max = 0.0
        
        for i, (psnr_data, ssim_data) in enumerate(zip(self.valid_psnr_data, self.valid_ssim_data)):
            current_psnr_min = psnr_data['min']
            current_psnr_max = psnr_data['max']
            current_ssim_min = ssim_data['min']
            current_ssim_max = ssim_data['max']
            
            if current_psnr_min < psnr_min:
                psnr_min = current_psnr_min
                psnr_min_indices = i
            
            if current_psnr_max > psnr_max:
                psnr_max = current_psnr_max
                psnr_max_indices = i
            
            if current_ssim_min < ssim_min:
                ssim_min = current_ssim_min
                ssim_min_indices = i
            
            if current_ssim_max > ssim_max:
                ssim_max = current_ssim_max
                ssim_max_indices = i
        
        print(f"# TRAIN RESULTs")        
        print(f"#\t PSNR(min/max) = {psnr_min:.3f} / {psnr_max:.3f} ({psnr_min_indices} / {psnr_max_indices})")
        print(f"#\t SSIM(min/max) = {ssim_min:.5f} / {ssim_max:.5f} ({ssim_min_indices} / {ssim_max_indices})")
                
        # Draw training_loss graph
        indices = []
        buffer_min = []
        # buffer_max = []
        buffer_avg = []
        for i, train_loss in enumerate(self.train_loss_data):
            indices.append(i)
            buffer_min.append(train_loss['min'])
            # buffer_max.append(train_loss['max'])
            buffer_avg.append(train_loss['avg'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(indices, buffer_min, 'r', label='Loss(min)')
        plt.plot(indices, buffer_avg, 'b', label='Loss(avg)')
        # plt.plot(indices, buffer_max, 'g', label='Loss(max)')
        plt.legend(loc='upper right')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.xticks(range(min(indices), max(indices)+1, 1), fontsize=5)
        
        plt.savefig(os.path.join(self.exp_dir, 'training_loss.png'))
        
        # Extract average value from list
        valid_psnr_avg_data = [item['avg'] for item in self.valid_psnr_data]
        valid_ssim_avg_data = [item['avg'] for item in self.valid_ssim_data]

        # Draw PSNR & SSIM graph
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # PSNR
        ax1.plot(range(min(indices), max(indices)+1, 1), valid_psnr_avg_data, 'r', label='PSNR')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR', color='r')

        # SSIM
        ax2 = ax1.twinx()
        ax2.plot(range(min(indices), max(indices)+1, 1), valid_ssim_avg_data, 'b', label='SSIM')
        ax2.set_ylabel('SSIM', color='b', rotation=-90, labelpad=15)

        ax1.set_xticks(range(min(indices), max(indices)+1, 1))
        ax1.set_xticklabels(range(min(indices), max(indices)+1, 1), fontsize=5)

        plt.title('Average of PSNR & SSIM')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.savefig(os.path.join(self.exp_dir, 'psnr_ssim_average.png'))

        elapsed_time = time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"# elapsed_time = {int(hours)}hr {int(minutes)}m {seconds:.1f}s")

        with open(os.path.join(self.exp_dir, 'summary.txt'), 'w') as f:
            f.write(f"# Elapsed_time = {int(hours)}hr {int(minutes)}m {seconds:.1f}s\n")
            f.write(f"# PSNR(min/max) = {psnr_min:.3f} / {psnr_max:.3f} ({psnr_min_indices} / {psnr_max_indices})\n")
            f.write(f"# SSIM(min/max) = {ssim_min:.5f} / {ssim_max:.5f} ({ssim_min_indices} / {ssim_max_indices})\n")
