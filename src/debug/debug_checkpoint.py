#!/usr/bin/env python3
"""
best_model.pth 로딩 검증 스크립트
"""

import torch
from iot_fed.task import Net

# best_model.pth 로드
checkpoint = torch.load('final_model.pt', map_location='cpu')
if 'model_state_dict' in checkpoint:
    best_state = checkpoint['model_state_dict']
else:
    best_state = checkpoint

print("=== best_model.pth 정보 ===")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Train Acc: {checkpoint.get('train_acc', 'N/A')}")
print(f"Val Acc: {checkpoint.get('val_acc', 'N/A')}")

print(f"\n=== best_model.pth 키 샘플 (첫 5개) ===")
for i, key in enumerate(list(best_state.keys())[:5]):
    print(f"  {key}")

# 빈 모델 생성
model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
model_state = model.state_dict()

print(f"\n=== 모델 구조 키 샘플 (첫 5개) ===")
for i, key in enumerate(list(model_state.keys())[:5]):
    print(f"  {key}")

# 키 형식 비교
best_has_model_prefix = any(k.startswith('model.') for k in best_state.keys())
model_has_model_prefix = any(k.startswith('model.') for k in model_state.keys())

print(f"\n=== 키 형식 분석 ===")
print(f"best_model.pth에 'model.' 접두사 있음: {best_has_model_prefix}")
print(f"Net 모델에 'model.' 접두사 있음: {model_has_model_prefix}")

# 키 변환 시뮬레이션 (server_app.py 로직 재현)
if not best_has_model_prefix:
    print("\n'model.' 접두사 추가 중...")
    converted_state = {f'model.{k}': v for k, v in best_state.items()}
else:
    converted_state = best_state

print(f"\n=== 변환 후 키 샘플 (첫 5개) ===")
for i, key in enumerate(list(converted_state.keys())[:5]):
    print(f"  {key}")

# 로딩 테스트
print("\n=== 로딩 테스트 ===")
try:
    model.load_state_dict(converted_state)
    print("로딩 성공!")
except Exception as e:
    print(f"로딩 실패: {e}")
    
    # 키 불일치 분석
    model_keys = set(model_state.keys())
    converted_keys = set(converted_state.keys())
    
    missing = model_keys - converted_keys
    unexpected = converted_keys - model_keys
    
    if missing:
        print(f"\n모델에는 있는데 checkpoint에 없는 키 ({len(missing)}개):")
        for key in list(missing)[:5]:
            print(f"  {key}")
    
    if unexpected:
        print(f"\ncheckpoint에는 있는데 모델에 없는 키 ({len(unexpected)}개):")
        for key in list(unexpected)[:5]:
            print(f"  {key}")
