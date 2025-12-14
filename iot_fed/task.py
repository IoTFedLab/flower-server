"""iot-fed: Flower / PyTorch 연합학습 애플리케이션"""

import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


class Net(nn.Module):
    """피부질환 분류를 위한 MobileNetV3-Small 모델 (6개 클래스)"""

    def __init__(self, num_classes=6, pretrained=False, drop_rate=0.2):
        super(Net, self).__init__()
        # MobileNetV3-Small 모델 생성
        self.model = timm.create_model(
            'mobilenetv3_small_100.lamb_in1k',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )

    def forward(self, x):
        return self.model(x)


fds = None  # FederatedDataset 캐시

# MobileNetV3용 변환 (224x224 입력 크기)
pytorch_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 통계
])


def apply_transforms(batch):
    """FederatedDataset 파티션에 변환 적용"""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, use_cifar10: bool = True, data_root: str = None, val_data_root: str = None):
    """
    연합학습을 위한 데이터 로드

    Args:
        partition_id: 현재 파티션 ID (CIFAR-10에서만 사용)
        num_partitions: 전체 파티션 수 (CIFAR-10에서만 사용)
        use_cifar10: True면 CIFAR-10 사용, False면 커스텀 피부질환 데이터셋 사용
        data_root: 훈련 데이터셋 루트 디렉토리 (use_cifar10=False일 때 필수)
        val_data_root: 검증 데이터셋 루트 디렉토리 (use_cifar10=False일 때 필수)

    Returns:
        (trainloader, valloader) 튜플
    """
    if use_cifar10:
        # CIFAR-10 데이터 로드 (원본 구현)
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
        valloader = DataLoader(partition_train_test["test"], batch_size=32)
    else:
        # 커스텀 피부질환 데이터 로드 (이미 train/val 분리된 상태)
        from iot_fed.dataset import load_skin_disease_data
        if data_root is None or val_data_root is None:
            raise ValueError("use_cifar10=False일 때 data_root와 val_data_root는 필수입니다")
        trainloader, valloader = load_skin_disease_data(
            data_root=data_root,
            val_data_root=val_data_root,
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=32
        )

    return trainloader, valloader


def train(net, trainloader, epochs, lr, device, global_params=None, mu=0.0):
    """학습 데이터셋으로 모델 훈련 (FedProx 지원)

    Args:
        net: 훈련할 모델
        trainloader: 훈련 데이터 로더
        epochs: 로컬 에폭 수
        lr: 학습률
        device: 디바이스 (CPU/GPU)
        global_params: 글로벌 모델 파라미터 (FedProx용, None이면 FedAvg)
        mu: FedProx proximal term 계수 (0.0이면 FedAvg와 동일)
    """
    net.to(device)  # GPU 사용 가능하면 모델을 GPU로 이동
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    net.train()
    running_loss = 0.0

    # FedProx: 글로벌 파라미터를 디바이스로 이동
    if global_params is not None and mu > 0:
        global_params = [p.to(device) for p in global_params]

    # 에폭별 진행 표시
    for epoch in range(epochs):
        epoch_loss = 0.0

        # 배치별 진행 표시
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False) as pbar:
            for batch in pbar:
                # dict (CIFAR-10)와 tuple (커스텀 데이터셋) 형식 모두 처리
                if isinstance(batch, dict):
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                # 기본 Cross Entropy Loss
                outputs = net(images)
                ce_loss = criterion(outputs, labels)

                # FedProx: Proximal term 추가
                if global_params is not None and mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(net.parameters(), global_params):
                        proximal_term += torch.norm(local_param - global_param, p=2) ** 2
                    loss = ce_loss + (mu / 2) * proximal_term
                else:
                    loss = ce_loss

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                running_loss += batch_loss
                epoch_loss += batch_loss

                # 실시간 loss 표시
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

        # 에폭 완료 후 평균 loss 출력
        avg_epoch_loss = epoch_loss / len(trainloader)
        print(f"   Epoch {epoch+1}/{epochs} 완료 - Avg Loss: {avg_epoch_loss:.4f}")

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device):
    """테스트 데이터셋으로 모델 검증"""
    net.to(device)
    net.eval()  # 평가 모드로 전환 (Dropout, BatchNorm 고정)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    with torch.no_grad():
        # 평가 진행 표시
        with tqdm(testloader, desc="평가 중", unit="batch", leave=False) as pbar:
            for batch in pbar:
                # dict (CIFAR-10)와 tuple (커스텀 데이터셋) 형식 모두 처리
                if isinstance(batch, dict):
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = net(images)
                batch_loss = criterion(outputs, labels).item()
                loss += batch_loss
                batch_correct = (torch.max(outputs.data, 1)[1] == labels).sum().item()
                correct += batch_correct

                # 실시간 정확도 표시
                current_acc = correct / ((pbar.n + 1) * len(labels))
                pbar.set_postfix({'acc': f'{current_acc:.4f}', 'loss': f'{batch_loss:.4f}'})

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    print(f"   평가 완료 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy
