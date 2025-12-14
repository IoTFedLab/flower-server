#!/usr/bin/env python3
"""
재현성 테스트: 같은 모델을 여러 번 평가해서 결과가 같은지 확인
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from iot_fed.task import Net, test as test_fn
from iot_fed.dataset import SkinDiseaseDataset

# 시드 고정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터셋 준비
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("재현성 테스트: 같은 모델을 5번 평가")
print("=" * 60)

checkpoint_path = 'final_model.pt'

results = []

for run in range(1, 6):
    print(f"\n{'='*60}")
    print(f"Run {run}/5")
    print(f"{'='*60}")
    
    # 시드 고정
    set_seed(42)
    
    # 모델 로드
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 키 변환
    if state_dict:
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith('model.'):
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 데이터셋 준비 (매번 새로 생성)
    val_dataset = SkinDiseaseDataset(
        data_root='data/validation',
        transform=val_transform
    )
    
    # Test 1: num_workers=0
    print(f"\n  Test 1: num_workers=0")
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    loss1, acc1 = test_fn(model, val_loader, device)
    print(f"    Loss: {loss1:.6f}, Acc: {acc1:.6f}")
    
    # Test 2: num_workers=2
    print(f"\n  Test 2: num_workers=2")
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    loss2, acc2 = test_fn(model, val_loader, device)
    print(f"    Loss: {loss2:.6f}, Acc: {acc2:.6f}")
    
    results.append({
        'run': run,
        'loss_nw0': loss1,
        'acc_nw0': acc1,
        'loss_nw2': loss2,
        'acc_nw2': acc2,
    })

print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)

print("\nnum_workers=0 결과:")
for r in results:
    print(f"  Run {r['run']}: Loss={r['loss_nw0']:.6f}, Acc={r['acc_nw0']:.6f}")

print("\nnum_workers=2 결과:")
for r in results:
    print(f"  Run {r['run']}: Loss={r['loss_nw2']:.6f}, Acc={r['acc_nw2']:.6f}")

# 분산 계산
acc_nw0_list = [r['acc_nw0'] for r in results]
acc_nw2_list = [r['acc_nw2'] for r in results]

print("\n" + "=" * 60)
print("통계")
print("=" * 60)
print(f"num_workers=0:")
print(f"  평균: {np.mean(acc_nw0_list):.6f}")
print(f"  표준편차: {np.std(acc_nw0_list):.6f}")
print(f"  최소-최대: {np.min(acc_nw0_list):.6f} - {np.max(acc_nw0_list):.6f}")

print(f"\nnum_workers=2:")
print(f"  평균: {np.mean(acc_nw2_list):.6f}")
print(f"  표준편차: {np.std(acc_nw2_list):.6f}")
print(f"  최소-최대: {np.min(acc_nw2_list):.6f} - {np.max(acc_nw2_list):.6f}")

if np.std(acc_nw0_list) < 0.0001:
    print("\nnum_workers=0: 완벽한 재현성")
else:
    print(f"\nnum_workers=0: 재현성 문제 (표준편차 {np.std(acc_nw0_list):.6f})")

if np.std(acc_nw2_list) < 0.0001:
    print("num_workers=2: 완벽한 재현성")
else:
    print(f"num_workers=2: 재현성 문제 (표준편차 {np.std(acc_nw2_list):.6f})")

print("=" * 60)
