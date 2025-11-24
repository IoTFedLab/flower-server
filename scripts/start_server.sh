#!/bin/bash
# 중앙 서버 시작 스크립트

echo "=========================================="
echo "🚀 Starting Federated Learning Server"
echo "=========================================="

# 서버 IP 확인
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "📍 Server IP: $SERVER_IP"
echo "📍 Port: 9092"
echo "📍 Binding: 0.0.0.0:9092 (모든 네트워크 인터페이스)"
echo ""
echo "⚠️  클라이언트는 다음 주소로 연결하세요:"
echo "   flwr-supernode --insecure --superlink $SERVER_IP:9092"
echo ""

# SuperLink 시작 (TCP Keepalive 설정 추가)
echo "🔗 Starting SuperLink with TCP Keepalive..."
# gRPC Keepalive 환경변수 설정
export GRPC_KEEPALIVE_TIME_MS=10000           # 10초마다 keepalive ping
export GRPC_KEEPALIVE_TIMEOUT_MS=5000         # 5초 응답 없으면 연결 끊김
export GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS=1  # 요청 없어도 keepalive 허용
export GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA=0    # ping 제한 없음

flower-superlink --insecure --isolation subprocess &
SUPERLINK_PID=$!

# SuperLink가 시작될 때까지 대기
sleep 3

echo "✅ SuperLink started (PID: $SUPERLINK_PID)"
echo ""
echo "📝 Next steps:"
echo "   1. Run ServerApp: flwr run ."
echo "   2. Connect clients to: $SERVER_IP:9092"
echo ""
echo "🛑 To stop: kill $SUPERLINK_PID"
echo "=========================================="

# 로그 출력
wait $SUPERLINK_PID

