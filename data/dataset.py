import torch
import os
import cv2
import random
import numpy as np
import albumentations as A          # DNN 성능을 향상시키는 컴퓨터 도구
from tqdm import tqdm               # 진행 상태(progress)를 보여주는 클래스
from typing import List, Tuple
from multiprocessing import Pool    # 각각의 프로세스에서 함수를 병렬로 실행할 수 있도록 해주는 클래스
from torch.utils.data import Dataset, DataLoader

"""
Original Color image -> HR : [256 x 256], LR : [64 x 64]
"""

# 입력값으로 np.ndarray 타입인 arr를 받고, torch.Tensor 타입을 리턴함을 명시
def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    입력된 NumPy 배열을 PyTorch tensor로 변환하고, 0~1 사이의 값을 갖도록 정규화합니다.

    Args:
        arr (numpy.ndarray): 변환할 NumPy 배열.

    Returns:
        torch.Tensor: 변환된 PyTorch tensor.

    Raises:
        TypeError: 입력이 numpy.ndarray가 아닌 경우 발생합니다.
        ValueError: 입력 배열의 shape이 (H, W, C) 또는 (B, H, W, C) 형태가 아닌 경우 발생합니다. (B = Batch)
    """
    # 입력이 np.ndarray이면 True, 아니면 False
    if not isinstance(arr, np.ndarray):
        # raise는 예외를 발생시켜서 함수의 실행을 중단하고 예외 처리를 수행
        raise TypeError(f"입력은 numpy.ndarray 형태여야 합니다. 입력 타입: {type(arr)}")

    # 입력 배열의 shape이 (H, W, C) 또는 (B, H, W, C) 형태인지 확인합니다.
    if len(arr.shape) != 3 and len(arr.shape) != 4:
        raise ValueError(f"입력 배열의 shape은 (H, W, C) 또는 (B, H, W, C) 형태여야 합니다. 입력 shape: {arr.shape}")

    # 배열 값을 0~1 사이로 정규화합니다.
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) #/ 255.0
    else:
        arr = arr.astype(np.float32)

    # axis 변환
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))      # (H, W, C) -> (C, H, W)
    else:
        arr = np.transpose(arr, (0, 3, 1, 2))   # (B, H, W, C) -> (B, C, H, W)

    # Numpy -> pytorch tensor로 변환
    return torch.from_numpy(arr)

# List[str] : str 타입의 원소를 가지는 리스트
def check_imgs(img_paths:List[str]):
    valid, error = 0, 0
    isblack = 0

    valid_img_path =[]
    
    bar = tqdm(img_paths, ncols=120)    # 반복할 리스트 = img_paths, ncols = progress bar의 너비
    
    for img_path in bar:
        obj = read_img(img_path)    # 한 장씩 이미지를 읽어옴
        if obj is None:             # 이미지를 못 읽어온 경우
            error += 1
        elif isinstance(obj, np.ndarray):   # obj가 np.ndarray의 인스턴스(객체)이면 True
            # read_img -> np.ndarray
            h, w, c = obj.shape
            if c == 1:      # 흑백 사진
                isblack += 1
            
            elif c == 3:    # 컬러 사진
                valid += 1
                valid_img_path.append(img_path)   # validation에 사용할 이미지의 경로를 넣음
        del obj # clear memory ... 
    
    print(f"Chekced imgs ... valid={valid}, io_error={error}, is_black={isblack}")
    
    return valid_img_path

# float=0.8, int=0 -> Default 값
def split_data(img_paths:List[str], r:float=0.8, seed:int=0):
    total_number = len(img_paths)               # 전체 이미지의 개수
    train_number = int(total_number * r)        # 80% -> train dataset
    valid_number = total_number - train_number  # 20% -> validation dataset
    
    # reproductable -> 계속
    random.seed(seed)
    
    # split
    train_img_paths = img_paths[:train_number]      # 0 ~ train_number
    valid_img_paths = img_paths[train_number:]      # train_number ~ end
    
    # print(f"# TRAIN_DATA = {train_number}, VALID_DATA = {valid_number}")
    
    return train_img_paths, valid_img_paths

# p : image path
def read_img(p:str):
    try:
        img = cv2.imread(p, cv2.IMREAD_COLOR)           # 컬러 이미지로 읽은 다음
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # RGB로 바꿈
        return img
    except:
        return None

# Dataset preprocessing & Preload 
class SR_Dataset(Dataset):
    # SR_Dataset의 객체가 생성될 때 자동으로 호출되는 Method
    def __init__(self, data_paths:list, preload:bool=False, train:bool=True, hr_size:int=256, lr_size:int=64, workers:int=8):
        super().__init__()          # 부모 클래스(Dataset) 호출
        self.length_of_data = len(data_paths)       # Dataset으로 사용할 이미지 개수
        self.workers = workers                      # 기본값 = 8
        self.preload = preload                      # 기본값 = False
        self.data_list = self._preload(data_paths, preload)     # preload = True이면 이미지들을 미리 Load 해놓겠다는 뜻
        # preload = True : data_list에는 Color & RGB 형태의 이미지 값들이 들어있다.
        # preload = False : data_list에는 이미지의 경로(data_paths)가 들어있다.
        
        # data augmentation
        # train이 bool 타입이고, True 일때
        if isinstance(train, bool) and train == True:  
            # Original -> High resolution [256 x 256]
            self.o2hr = A.Compose([
                A.RandomCrop(height=hr_size, width=hr_size),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(limit=45, interpolation=cv2.INTER_CUBIC),
            ])
            # High resolution -> Low resolution [64 x 64]
            self.hr2lr = A.Compose([
                A.AdvancedBlur(p=0.5),                                                              # 블러 현상 추가
                A.Downscale(scale_min=0.7, scale_max=0.95, p=0.5, interpolation=cv2.INTER_LINEAR),  # 다운 스케일링
                A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),                                     # 노이즈 생성
                A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),                     # 이미지 압축
                A.AdvancedBlur(p=0.5),                                                              
                A.Resize(height=lr_size, width=lr_size),                                            # 사이즈 재조정
                A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),
                A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5)
            ])
        # train이 bool 타입이고, False 일때
        elif isinstance(train, bool) and train == False:
            self.o2hr = A.CenterCrop(height=hr_size, width=hr_size)
            self.hr2lr = A.Resize(height=lr_size, width=lr_size)
        else:
            raise RuntimeError(train)
    
    # RAM에 이미 파일들을 바이너리 형태로 미리 Load -> 이미지 읽어오는 속도 빨라짐
    def _preload(self, data_paths:list, preload:bool=False):
        if not preload:     # preload가 False인 경우 data_paths 반환
            return data_paths
        else:   # preload = True
            with Pool(processes=self.workers) as pool:      # 최대 8개(self.workers)의 프로세스를 병렬로 처리, with .. as.. = 문단속
                preloaded_data = []
                # 16개씩 이미지를 read해서 img_data에 넣음
                # img_data에는 Color & RGB 형태의 이미지 값들이 저장됨
                for img_data in tqdm(pool.imap_unordered(read_img, data_paths, chunksize=16), total=self.length_of_data):
                    preloaded_data.append(img_data)
            
            pool.close()    # 지금 수행 중인 작업이 모두 끝나면 Pool의 프로세스들을 종료
            pool.join()     # Pool의 모든 프로세스가 종료될 때까지 기다림

            print("\033[F\033[J ==> IMAGE PRELOAD COMPLETE")
            return preloaded_data

    # Dataset으로 사용할 이미지 개수 반환
    def __len__(self) -> int:
        return self.length_of_data
    
    # Tensor 형태의 값을 가지는 Tuple 반환
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:    # preload = True -> _preload 메서드에서 이미지를 미리 Load & read 하여 data_list에 올려놓음
            orig_obj = self.data_list[idx]
        else:               # preload = False -> data_paths로부터 이미지를 읽어옴
            orig_obj = read_img(self.data_list[idx])
        
        # o2hr, hr2lr 모두 key가 하나인 딕셔너리를 반환 -> key = "image"
        hr_obj = self.o2hr(image=orig_obj)["image"]
        lr_obj = self.hr2lr(image=hr_obj)["image"]
        
        hr_tensor = to_tensor(hr_obj)
        lr_tensor = to_tensor(lr_obj)
        
        return lr_tensor, hr_tensor


def get_dataloader(root:str, check:bool=False, split_ratio:float=0.8, preload:bool=False, batch_size:int=32, workers:int=8,
                   hr_size:int=256, lr_size:int=64):
    # root 폴더 안에 있는 파일들을 x에 리스트 형태로 올리고, os.path.join 함수를 이용해 각 이미지의 경로들을 total_img_path에 저장
    total_img_paths = [os.path.join(root, x) for x in os.listdir(root)]

    # Check data
    if check:
        imgs_list = check_imgs(total_img_paths)
    else:
        imgs_list = total_img_paths

    # Split data
    train_imgs_list, valid_imgs_list = split_data(imgs_list, split_ratio)

    # build Dataset
    train_dataset = SR_Dataset(train_imgs_list, preload, True, hr_size, lr_size, workers)
    valid_dataset = SR_Dataset(valid_imgs_list, preload, False, hr_size, lr_size, workers)
    
    # build DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=workers, persistent_workers=True)
    
    return train_loader, valid_loader
