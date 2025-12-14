"""iot-fed: Flower / PyTorch 연합학습 서버 애플리케이션"""

import torch
import time
from pathlib import Path
from tqdm import tqdm
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from iot_fed.task import Net


# 라운드별 모델 저장을 위한 콜백 함수
def save_round_model(round_num: int, arrays: ArrayRecord, save_dir: Path):
    """라운드별 모델 저장"""
    state_dict = arrays.to_torch_state_dict()
    model_path = save_dir / f"round_{round_num}.pt"
    torch.save(state_dict, model_path)
    print(f"   Round {round_num} 모델 저장: {model_path}")

# ServerApp 생성
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """ServerApp의 메인 진입점"""

    # 실행 설정 읽기
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    mu: float = context.run_config.get("fedprox-mu", 0.0)  # FedProx mu 파라미터
    checkpoint_path: str = context.run_config.get("checkpoint-path", None)
    resume_from_final: bool = context.run_config.get("resume-from-final", False)

    # 글로벌 모델 로드
    global_model = Net(num_classes=6, pretrained=False, drop_rate=0.2)

    # 이어서 학습 여부 확인
    final_model_path = Path("final_model.pt")
    if resume_from_final and final_model_path.exists():
        print(f"\n이전 연합학습 결과에서 이어서 학습: {final_model_path}")
        checkpoint = torch.load(final_model_path, map_location='cpu')
        state_dict = checkpoint

        # state_dict 키 형식 확인 (이미 'model.' 접두사가 있을 것)
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith('model.'):
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}

        global_model.load_state_dict(state_dict)
        print("   이전 연합학습 모델 로드 성공!\n")
    # 체크포인트 경로가 제공되면 사전학습된 가중치 로드
    elif checkpoint_path and Path(checkpoint_path).exists():
        print(f"\n사전학습된 가중치 로드 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 다양한 체크포인트 형식 처리
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            train_acc = checkpoint.get('train_acc', 'N/A')
            val_acc = checkpoint.get('val_acc', 'N/A')

            print(f"   Epoch {epoch}에서 체크포인트 로드됨")
            if isinstance(train_acc, (int, float)):
                print(f"   Train Acc: {train_acc:.2%}")
            else:
                print(f"   Train Acc: {train_acc}")
            if isinstance(val_acc, (int, float)):
                print(f"   Val Acc: {val_acc:.2%}")
            else:
                print(f"   Val Acc: {val_acc}")
        else:
            state_dict = checkpoint

        # state_dict 키 형식 확인 및 변환
        # 체크포인트가 'conv_stem.weight' 형식이면 'model.conv_stem.weight'로 변환
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith('model.'):
            print("   state_dict 키에 'model.' 접두사 추가 중...")
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}

        global_model.load_state_dict(state_dict)
        print("   사전학습된 가중치 로드 성공!\n")
    else:
        print("\n체크포인트가 제공되지 않았거나 파일을 찾을 수 없습니다. 랜덤 가중치로 시작합니다.\n")

    arrays = ArrayRecord(global_model.state_dict())

    # 라운드별 모델 저장 디렉토리 생성
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"라운드별 모델 저장 디렉토리: {models_dir}\n")

    # FedAvg 전략 초기화
    strategy = FedAvg(
        fraction_train=fraction_train,
        min_train_nodes=3,       # 학습에 최소 3개 노드 필요
        min_evaluate_nodes=3,    # 평가에 최소 3개 노드 필요
        min_available_nodes=3,   # 시작에 최소 3개 노드 필요 (이 수만큼 연결될 때까지 대기)
    )

    # 각 라운드마다 수동으로 실행하며 모델 저장
    current_arrays = arrays

    print(f"\n{'='*70}")
    print(f"연합학습 시작!")
    print(f"{'='*70}")
    print(f"설정:")
    print(f"   - 총 라운드: {num_rounds}")
    print(f"   - 로컬 에폭: {context.run_config['local-epochs']}")
    print(f"   - 학습률: {lr}")
    print(f"   - FedProx mu: {mu} {'(FedProx 활성화)' if mu > 0 else '(FedAvg 모드)'}")
    print(f"   - 클라이언트 참여 비율: {fraction_train}")
    print(f"{'='*70}\n")

    # 전체 학습 시작 시간
    total_start_time = time.time()

    # tqdm 진행 바 생성
    with tqdm(total=num_rounds, desc="연합학습 진행", unit="round", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for round_num in range(1, num_rounds + 1):
            round_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'='*60}")

            # 1 라운드만 실행 (FedProx mu 파라미터 포함)
            result = strategy.start(
                grid=grid,
                initial_arrays=current_arrays,
                train_config=ConfigRecord({"lr": lr, "mu": mu}),
                num_rounds=1,
            )

            # 라운드 결과 저장
            current_arrays = result.arrays
            save_round_model(round_num, current_arrays, models_dir)

            # 라운드 소요 시간 계산
            round_elapsed = time.time() - round_start_time
            total_elapsed = time.time() - total_start_time
            avg_time_per_round = total_elapsed / round_num
            remaining_rounds = num_rounds - round_num
            eta = avg_time_per_round * remaining_rounds

            print(f"   Round {round_num} 소요 시간: {round_elapsed:.2f}초")
            if remaining_rounds > 0:
                print(f"   예상 남은 시간: {eta:.2f}초 ({eta/60:.1f}분)")

            # tqdm 업데이트
            pbar.set_postfix({
                'Round': f'{round_elapsed:.1f}s',
                'Avg': f'{avg_time_per_round:.1f}s',
                'ETA': f'{eta/60:.1f}m' if remaining_rounds > 0 else 'Done'
            })
            pbar.update(1)

    # 전체 소요 시간 출력
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"전체 연합학습 완료!")
    print(f"   총 소요 시간: {total_elapsed:.2f}초 ({total_elapsed/60:.1f}분)")
    print(f"   평균 라운드 시간: {total_elapsed/num_rounds:.2f}초")
    print(f"{'='*70}\n")

    # 최종 모델을 디스크에 저장
    print(f"\n{'='*60}")
    print("최종 모델을 디스크에 저장 중...")
    state_dict = current_arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("final_model.pt 저장 완료")

    # 저장된 모델 목록 출력
    print(f"\n저장된 모델 파일:")
    print(f"   - final_model.pt (최종 모델)")
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("round_*.pt")):
            print(f"   - {model_file}")
    print(f"{'='*60}\n")
