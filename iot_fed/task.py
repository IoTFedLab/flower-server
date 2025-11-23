"""iot-fed: Flower / PyTorch 연합학습 애플리케이션"""

import torch
import torch.nn as nn
import timm
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


def train(net, trainloader, epochs, lr, device):
    """학습 데이터셋으로 모델 훈련"""
    net.to(device)  # GPU 사용 가능하면 모델을 GPU로 이동
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            # dict (CIFAR-10)와 tuple (커스텀 데이터셋) 형식 모두 처리
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """테스트 데이터셋으로 모델 검증"""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            # dict (CIFAR-10)와 tuple (커스텀 데이터셋) 형식 모두 처리
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
