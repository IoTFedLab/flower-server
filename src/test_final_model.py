#!/usr/bin/env python3
"""
연합학습 최종 모델 테스트 스크립트
"""

import torch
from iot_fed.task import Net, test_fn
from iot_fed.dataset import load_skin_disease_data


def test_final_model():
    """final_model.pt 성능 테스트"""
    
    print("=" * 60)
    print("🧪 연합학습 최종 모델 테스트")
    print("=" * 60)
    
    # 1. 모델 로드
    print("\n1️⃣ 모델 로드 중...")
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
    
    try:
        state_dict = torch.load('final_model.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        print("   ✅ final_model.pt 로드 성공")
    except FileNotFoundError:
        print("   ❌ final_model.pt 파일을 찾을 수 없습니다.")
        print("   💡 먼저 연합학습을 실행하세요: flwr run .")
        return
    
    # 2. 테스트 데이터 로드
    print("\n2️⃣ 테스트 데이터 로드 중...")
    _, testloader = load_skin_disease_data(
        data_root='data/train',
        val_data_root='data/validation',
        partition_id=0,
        num_partitions=1,
        batch_size=32,
        test_split=0.2
    )
    print(f"   ✅ 테스트 샘플 수: {len(testloader.dataset)}")
    
    # 3. 평가
    print("\n3️⃣ 모델 평가 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  디바이스: {device}")
    
    test_loss, test_acc = test_fn(model, testloader, device)
    
    # 4. 결과 출력
    print("\n" + "=" * 60)
    print("📊 최종 결과")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print("=" * 60)
    
    # 5. 비교 (best_model.pth와 비교)
    print("\n5️⃣ 사전학습 모델과 비교...")
    try:
        checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
        pretrained_acc = checkpoint.get('val_acc', 'N/A')
        
        print(f"\n📈 성능 비교:")
        print(f"   사전학습 모델 (best_model.pth):  Val Acc = {pretrained_acc}")
        print(f"   연합학습 모델 (final_model.pt):  Test Acc = {test_acc:.2%}")
        
        if isinstance(pretrained_acc, (int, float)) and test_acc > pretrained_acc:
            print(f"\n   🎉 연합학습으로 {(test_acc - pretrained_acc):.2%} 향상!")
        elif isinstance(pretrained_acc, (int, float)):
            print(f"\n   ⚠️  연합학습 후 {(pretrained_acc - test_acc):.2%} 감소")
            print("   💡 더 많은 라운드나 epoch이 필요할 수 있습니다.")
    except FileNotFoundError:
        print("   ⚠️  best_model.pth를 찾을 수 없어 비교를 건너뜁니다.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_final_model()

