#!/usr/bin/env python3
"""
Round 1 모델과 best_model.pth 비교
"""

import torch

# best_model.pth 로드
best_checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
if 'model_state_dict' in best_checkpoint:
    best_state = best_checkpoint['model_state_dict']
else:
    best_state = best_checkpoint

# 키 변환
if not any(k.startswith('model.') for k in best_state.keys()):
    best_state = {f'model.{k}': v for k, v in best_state.items()}

# Round 1 모델 로드
try:
    round1_state = torch.load('models/round_1.pt', map_location='cpu')
    
    print("=== Round 1 vs best_model.pth 비교 ===")
    
    # 가중치 차이 계산
    total_diff = 0
    count = 0
    for key in list(best_state.keys())[:10]:
        diff = (round1_state[key] - best_state[key]).abs().mean().item()
        total_diff += diff
        count += 1
        print(f"{key[:50]:50s}: 차이 = {diff:.6f}")
    
    avg_diff = total_diff / count
    print(f"\n평균 차이 (첫 10개 레이어): {avg_diff:.6f}")
    
    if avg_diff < 0.0001:
        print("❌ Round 1에서 거의 학습 안 됨 → lr이 너무 낮거나 데이터 부족")
    elif avg_diff > 1.0:
        print("❌ Round 1에서 가중치가 너무 많이 변함 → lr이 너무 높음")
    else:
        print("✅ Round 1 학습이 정상적으로 진행됨")
    
except FileNotFoundError:
    print("❌ models/round_1.pt 파일을 찾을 수 없습니다.")
    print("   연합학습을 실행했는지 확인하세요.")

