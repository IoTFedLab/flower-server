#!/usr/bin/env python3
"""
final_model.pt 검증 스크립트
"""

import torch
from iot_fed.task import Net

# 1. final_model.pt 로드
final_state = torch.load('final_model.pt', map_location='cpu')

print("=== final_model.pt 키 샘플 (첫 5개) ===")
for i, key in enumerate(list(final_state.keys())[:5]):
    print(f"  {key}")

# 2. 모델에 로딩 테스트
model = Net(num_classes=6, pretrained=False, drop_rate=0.2)

print("\n=== 로딩 테스트 ===")
try:
    model.load_state_dict(final_state)
    print("✅ 로딩 성공!")
except Exception as e:
    print(f"❌ 로딩 실패: {e}")

# 3. 가중치 통계 (랜덤인지 학습된 건지 확인)
print("\n=== 가중치 통계 ===")
sample_key = list(final_state.keys())[0]
sample_weight = final_state[sample_key]
print(f"샘플 키: {sample_key}")
print(f"Shape: {sample_weight.shape}")
print(f"Mean: {sample_weight.mean():.6f}")
print(f"Std: {sample_weight.std():.6f}")
print(f"Min: {sample_weight.min():.6f}")
print(f"Max: {sample_weight.max():.6f}")

# 랜덤 초기화와 비교
model_random = Net(num_classes=6, pretrained=False, drop_rate=0.2)
random_weight = model_random.state_dict()[sample_key]
print(f"\n랜덤 초기화 가중치 (비교용):")
print(f"Mean: {random_weight.mean():.6f}")
print(f"Std: {random_weight.std():.6f}")

