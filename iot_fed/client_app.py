"""iot-fed: Flower / PyTorch 연합학습 클라이언트 애플리케이션"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from iot_fed.task import Net, load_data
from iot_fed.task import test as test_fn
from iot_fed.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """로컬 데이터로 모델 훈련"""

    # 모델 로드 및 수신한 가중치로 초기화
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 데이터 로드
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    use_cifar10 = context.run_config.get("use-cifar10", True)
    data_root = context.run_config.get("data-root", None)
    val_data_root = context.run_config.get("val-data-root", None)

    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        use_cifar10=use_cifar10,
        data_root=data_root,
        val_data_root=val_data_root
    )

    # 훈련 함수 호출
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # 응답 메시지 구성 및 반환
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """로컬 데이터로 모델 평가"""

    # 모델 로드 및 수신한 가중치로 초기화
    model = Net(num_classes=6, pretrained=False, drop_rate=0.2)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 데이터 로드
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    use_cifar10 = context.run_config.get("use-cifar10", True)
    data_root = context.run_config.get("data-root", None)
    val_data_root = context.run_config.get("val-data-root", None)

    _, valloader = load_data(
        partition_id,
        num_partitions,
        use_cifar10=use_cifar10,
        data_root=data_root,
        val_data_root=val_data_root
    )

    # 평가 함수 호출
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # 응답 메시지 구성 및 반환
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
