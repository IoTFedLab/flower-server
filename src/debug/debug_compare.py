#!/usr/bin/env python3
"""
best_model.pth vs final_model.pt 비교 스크립트
"""

import torch

# 로드
best_checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
if 'model_state_dict' in best_checkpoint:
    best_state = best_checkpoint['model_state_dict']
else:
    best_state = best_checkpoint

final_state = torch.load('final_model.pt', map_location='cpu')

# 키 변환 (server_app.py 로직 재현)
if not any(k.startswith('model.') for k in best_state.keys()):
    best_state = {f'model.{k}': v for k, v in best_state.items()}

print("=== 키 개수 비교 ===")
print(f"best_model.pth: {len(best_state)} keys")
print(f"final_model.pt: {len(final_state)} keys")

print("\n=== 키 일치 여부 ===")
best_keys = set(best_state.keys())
final_keys = set(final_state.keys())

if best_keys == final_keys:
    print("✅ 키가 완전히 일치")
    
    # 가중치 값 비교
    print("\n=== 가중치 차이 분석 (첫 10개 레이어) ===")
    total_diff = 0
    count = 0
    for key in list(best_keys)[:10]:
        diff = (final_state[key] - best_state[key]).abs().mean().item()
        total_diff += diff
        count += 1
        print(f"{key[:50]:50s}: 평균 차이 = {diff:.6f}")
    
    avg_diff = total_diff / count
    print(f"\n평균 차이 (첫 {count}개 레이어): {avg_diff:.6f}")
    
    if avg_diff < 0.0001:
        print("⚠️  가중치가 거의 동일 → 학습이 거의 안 된 것일 수 있음")
    elif avg_diff < 0.01:
        print("⚠️  가중치 변화가 매우 작음 → 학습이 조금만 진행됨")
    else:
        print("✅ 가중치가 충분히 변경됨 → 학습은 진행됨")
    
    # 전체 레이어 평균 차이
    print("\n=== 전체 레이어 평균 차이 ===")
    all_diffs = []
    for key in best_keys:
        diff = (final_state[key] - best_state[key]).abs().mean().item()
        all_diffs.append(diff)
    
    import statistics
    print(f"평균: {statistics.mean(all_diffs):.6f}")
    print(f"중앙값: {statistics.median(all_diffs):.6f}")
    print(f"최대: {max(all_diffs):.6f}")
    print(f"최소: {min(all_diffs):.6f}")
    
else:
    print("❌ 키가 불일치")
    only_in_best = best_keys - final_keys
    only_in_final = final_keys - best_keys
    
    if only_in_best:
        print(f"\nbest에만 있는 키 ({len(only_in_best)}개):")
        for key in list(only_in_best)[:5]:
            print(f"  {key}")
    
    if only_in_final:
        print(f"\nfinal에만 있는 키 ({len(only_in_final)}개):")
        for key in list(only_in_final)[:5]:
            print(f"  {key}")

