#!/usr/bin/env python3
"""
각 라운드별 모델 성능 평가
"""

import torch
from pathlib import Path
from iot_fed.task import Net, test as test_fn
from iot_fed.dataset import SkinDiseaseDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# 데이터셋 준비
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = SkinDiseaseDataset(
    data_root='data/validation',
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0  # 재현성을 위해 0으로 설정
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("각 라운드별 성능 평가")
print("=" * 60)

# models 디렉토리의 모든 round_*.pt 파일 찾기
models_dir = Path('models')
if not models_dir.exists():
    print("models/ 디렉토리가 없습니다.")
    exit(1)

round_files = sorted(models_dir.glob('round_*.pt'))

if not round_files:
    print("round_*.pt 파일을 찾을 수 없습니다.")
    exit(1)

results = []

for round_file in round_files:
    round_num = int(round_file.stem.split('_')[1])

    # 모델 로드
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
    state_dict = torch.load(round_file, map_location='cpu')

    # 키 형식 변환 (필요시)
    # models/round_*.pt가 'conv_stem.weight' 형식이면 'model.conv_stem.weight'로 변환
    if state_dict:
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith('model.'):
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # 평가
    loss, acc = test_fn(model, val_loader, device)
    results.append((round_num, loss, acc))

    print(f"Round {round_num:2d}: Loss = {loss:.4f}, Acc = {acc:.2%}")

print("\n" + "=" * 60)
print("성능 추이 분석")
print("=" * 60)

# 최고/최저 성능
best_round = max(results, key=lambda x: x[2])
worst_round = min(results, key=lambda x: x[2])

print(f"최고 성능: Round {best_round[0]} - {best_round[2]:.2%}")
print(f"최저 성능: Round {worst_round[0]} - {worst_round[2]:.2%}")

# 추이 분석
if len(results) >= 3:
    first_acc = results[0][2]
    mid_acc = results[len(results)//2][2]
    last_acc = results[-1][2]
    
    print(f"\n추이:")
    print(f"  초반 (Round {results[0][0]}): {first_acc:.2%}")
    print(f"  중반 (Round {results[len(results)//2][0]}): {mid_acc:.2%}")
    print(f"  후반 (Round {results[-1][0]}): {last_acc:.2%}")
    
    if last_acc < first_acc:
        print("\n성능이 떨어졌습니다 → lr이 너무 높거나 과적합")
    elif last_acc - first_acc < 0.01:
        print("\n성능 향상이 거의 없습니다 → lr이 너무 낮거나 데이터 부족")
    else:
        print("\n성능이 향상되었습니다")

print("=" * 60)
