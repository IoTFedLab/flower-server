#!/usr/bin/env python3
"""
연합학습 최종 모델 테스트 스크립트 (상세 평가 버전)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from iot_fed.task import Net
from iot_fed.dataset import SkinDiseaseDataset


def load_model(checkpoint_path: str, device: torch.device):
    """
    체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스

    Returns:
        model: 로드된 모델
        checkpoint_info: 체크포인트 메타데이터
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 생성
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)

    # 가중치 로드
    checkpoint_info = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_acc': checkpoint.get('train_acc', 'N/A'),
            'val_acc': checkpoint.get('val_acc', 'N/A')
        }
    else:
        # 직접 state_dict인 경우 (final_model.pt 등)
        state_dict = checkpoint

    # 키 형식 변환 (필요시)
    if state_dict:
        first_key = next(iter(state_dict.keys()))
        # best_model.pth: 'conv_stem.weight' → 'model.conv_stem.weight' 변환 필요
        if not first_key.startswith('model.'):
            print("   ℹ️  state_dict 키에 'model.' 접두사 추가 중...")
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}
        # final_model.pt: 'model.conv_stem.weight' → 그대로 사용

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("   ✅ 모델 로드 성공")
    if checkpoint_info:
        if checkpoint_info['epoch'] != 'N/A':
            print(f"   ✓ Epoch: {checkpoint_info['epoch']}")
        if checkpoint_info['train_acc'] != 'N/A':
            print(f"   ✓ Train Acc: {checkpoint_info['train_acc']:.2f}%")
        if checkpoint_info['val_acc'] != 'N/A':
            print(f"   ✓ Val Acc: {checkpoint_info['val_acc']:.2f}%")

    return model, checkpoint_info


def evaluate(model, dataloader, criterion, device, class_names):
    """
    모델 평가 (상세 버전)

    Args:
        model: 평가할 모델
        dataloader: Validation DataLoader
        criterion: Loss 함수
        device: 디바이스
        class_names: 클래스 이름 리스트

    Returns:
        dict: 평가 결과
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    print("Starting evaluation...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="🔍 평가 중", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Progress bar 업데이트
            current_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': f'{current_acc:.4f}'})

    # 전체 통계
    val_loss = running_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    # 클래스별 정확도
    per_class_acc = {}
    for idx, class_name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() > 0:
            class_acc = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc[class_name] = {
                'accuracy': class_acc,
                'correct': int((all_preds[mask] == all_labels[mask]).sum()),
                'total': int(mask.sum())
            }

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': val_loss,
        'accuracy': accuracy,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_results(results, class_names):
    """
    평가 결과 출력

    Args:
        results (dict): 평가 결과
        class_names (list): 클래스 이름 리스트
    """
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Val Loss: {results['loss']:.4f}")
    print(f"Val Accuracy: {results['accuracy']:.2f}%")

    print("\n" + "-"*60)
    print("Per-class Accuracy:")
    print("-"*60)
    for class_name, stats in results['per_class_acc'].items():
        print(f"  {class_name:8s}: {stats['accuracy']:5.1f}% ({stats['correct']:3d}/{stats['total']:3d})")

    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    print("Rows: True labels, Columns: Predicted labels")
    print(f"Classes: {', '.join(class_names)}")
    print()
    cm = results['confusion_matrix']

    # 헤더 출력
    header = "       " + "  ".join([f"{name[:4]:>4s}" for name in class_names])
    print(header)
    print("-" * len(header))

    # 각 행 출력
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:6]:6s} " + "  ".join([f"{cm[i][j]:4d}" for j in range(len(class_names))])
        print(row)

    print("="*60)


def validate_one_model(model_path: str):
    """메인 함수"""
    print("=" * 60)
    print("🧪 연합학습 최종 모델 테스트")
    print("=" * 60)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  디바이스: {device}")

    # NOTE: 모델 로드
    print("\n1️⃣ 모델 로드 중...")
    checkpoint_path = model_path
    model, checkpoint_info = load_model(checkpoint_path, device)

    # 2. 데이터셋 준비
    print("\n2️⃣ 데이터셋 준비 중...")
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
        num_workers=0,  # 재현성을 위해 0으로 설정
        pin_memory=False
    )

    print(f"   ✅ Validation 샘플 수: {len(val_dataset)}")

    # 클래스 이름 추출
    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"   ✅ 클래스: {', '.join(class_names)}")

    # 3. 평가
    print("\n3️⃣ 모델 평가 중...")
    criterion = nn.CrossEntropyLoss()
    results = evaluate(model, val_loader, criterion, device, class_names)

    # 4. 결과 출력
    print_results(results, class_names)

