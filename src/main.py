from src.test.test_final_model import validate_one_model
# from src.test.test_all_rounds import test_all_rounds

if __name__ == "__main__":
    # 하나의 모델 성능 평가
    num = 6
    validate_one_model(f'models/round_{num}.pt')
    
    # 모든 라운드별 성능 평가
    # test_all_rounds()