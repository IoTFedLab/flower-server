#!/bin/bash

# 연합학습 서버 방화벽 설정 스크립트
# 특정 클라이언트 IP만 9092 포트 접근 허용

# 클라이언트 IP 주소
CLIENT1_IP=""  # 시우
CLIENT2_IP=""  # 민지
CLIENT3_IP=""  # 소연
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "연합학습 서버 방화벽 설정"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "설정된 클라이언트 IP:"
echo "  $CLIENT1_IP"
echo "  $CLIENT2_IP"
echo "  $CLIENT3_IP"
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "방화벽 설정 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# UFW 설치 확인
echo "UFW 설치 확인 중..."
if ! command -v ufw &> /dev/null; then
    echo "   UFW가 설치되지 않았습니다. 설치 중..."
    sudo apt update
    sudo apt install -y ufw
    echo "   UFW 설치 완료"
else
    echo "   UFW 이미 설치됨"
fi

echo ""
echo "기존 9092 포트 규칙 삭제 중..."
# 9092 포트 관련 규칙이 있는지 확인
if sudo ufw status numbered | grep -q "9092"; then
    echo "   삭제할 규칙:"
    sudo ufw status numbered | grep "9092"
    echo ""
    # 역순으로 삭제 (번호가 바뀌지 않도록)
    for num in $(sudo ufw status numbered | grep "9092" | awk '{print $1}' | sed 's/\[//;s/\]//' | sort -rn); do
        echo "   규칙 [$num] 삭제 중..."
        echo "y" | sudo ufw delete $num
    done
    echo "   기존 9092 규칙 삭제 완료"
else
    echo "   삭제할 9092 규칙 없음"
fi

echo ""
echo "SSH 포트 허용 중 (중요!)..."
sudo ufw allow 22/tcp
echo "   SSH (22/tcp) 허용됨"

echo ""
echo "클라이언트 IP만 9092 포트 접근 허용 중..."

# 클라이언트 1 허용
sudo ufw allow from $CLIENT1_IP to any port 9092 proto tcp
echo "   클라이언트 1 ($CLIENT1_IP) 허용"

# 클라이언트 2 허용
sudo ufw allow from $CLIENT2_IP to any port 9092 proto tcp
echo "   클라이언트 2 ($CLIENT2_IP) 허용"

# 클라이언트 3 허용
# sudo ufw allow from $CLIENT3_IP to any port 9092 proto tcp
# echo "   클라이언트 3 ($CLIENT3_IP) 허용"

echo ""
echo "방화벽 활성화 중..."
sudo ufw --force enable
echo "   방화벽 활성화 완료"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "최종 방화벽 규칙:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo ufw status numbered
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "방화벽 설정 완료!"
echo ""

