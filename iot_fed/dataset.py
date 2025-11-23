"""피부질환 연합학습을 위한 커스텀 데이터셋 로더"""

import os
import unicodedata
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SkinDiseaseDataset(Dataset):
    """
    연합학습을 위한 피부질환 데이터셋

    디렉토리 구조:
        data_root/
        ├── 건선/
        │   ├── 건선_정면/[instance_id]/[id].png
        │   └── 건선_측면/[instance_id]/[id].png
        ├── 아토피/
        │   ├── 아토피_정면/[instance_id]/[id].png
        │   └── 아토피_측면/[instance_id]/[id].png
        ...

    클래스: 건선, 아토피, 여드름, 정상, 주사, 지루 (6개 클래스)
    """

    def __init__(self, data_root: str, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {
            '건선': 0,
            '아토피': 1,
            '여드름': 2,
            '정상': 3,
            '주사': 4,
            '지루': 5
        }

        self._load_samples()

    def _load_samples(self):
        """모든 이미지 경로와 레이블 로드"""
        # 질환 디렉토리 순회 (건선, 아토피, 등)
        for disease_dir in self.data_root.iterdir():
            if not disease_dir.is_dir():
                continue

            # 유니코드 정규화 (NFD → NFC 변환)
            disease_name = unicodedata.normalize('NFC', disease_dir.name)
            if disease_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[disease_name]

            # 방향 디렉토리 순회 (건선_정면, 건선_측면, 등)
            for direction_dir in disease_dir.iterdir():
                if not direction_dir.is_dir():
                    continue

                # 인스턴스 디렉토리 순회
                for instance_dir in direction_dir.iterdir():
                    if not instance_dir.is_dir():
                        continue

                    # 모든 PNG 이미지 찾기
                    for img_path in instance_dir.glob('*.png'):
                        self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_skin_disease_data(
    data_root: str,
    val_data_root: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    test_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    연합학습을 위한 피부질환 데이터 로드

    Args:
        data_root: 훈련 데이터셋 루트 디렉토리
        val_data_root: 검증 데이터셋 루트 디렉토리
        partition_id: 현재 파티션 ID (사용 안 함 - 각 클라이언트에 이미 분할된 데이터 배포됨)
        num_partitions: 전체 파티션 수 (사용 안 함)
        batch_size: DataLoader 배치 크기
        test_split: 사용 안 함 (이미 train/val 분리됨)

    Returns:
        (train_loader, val_loader) 튜플
    """
    # 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 훈련 데이터 로드 (이미 분할되어 배포된 상태)
    train_dataset = SkinDiseaseDataset(data_root, transform=transform)

    # 검증 데이터 로드 (이미 분할되어 배포된 상태)
    val_dataset = SkinDiseaseDataset(val_data_root, transform=transform)

    # DataLoader 생성 (파티션 분할 없이 전체 데이터 사용)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

