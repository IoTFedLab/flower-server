#!/bin/bash
# 중앙 서버 시작 스크립트

echo "=========================================="
echo "🚀 Starting Federated Learning Server"
echo "=========================================="

# 서버 IP 확인
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "📍 Server IP: $SERVER_IP"
echo "📍 Port: 9092"
echo ""

# SuperLink 시작
echo "🔗 Starting SuperLink..."
flwr-superlink --insecure &
SUPERLINK_PID=$!

# SuperLink가 시작될 때까지 대기
sleep 3

echo "✅ SuperLink started (PID: $SUPERLINK_PID)"
echo ""
echo "📝 Next steps:"
echo "   1. Run ServerApp: flwr-serverapp iot_fed.server_app:app --insecure"
echo "   2. Connect clients to: $SERVER_IP:9092"
echo ""
echo "🛑 To stop: kill $SUPERLINK_PID"
echo "=========================================="

# 로그 출력
wait $SUPERLINK_PID

