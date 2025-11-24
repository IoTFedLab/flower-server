#!/bin/bash
# 클라이언트 시작 스크립트

if [ -z "$1" ]; then
    echo "Usage: ./start_client.sh <SERVER_IP>"
    echo "Example: ./start_client.sh 192.168.1.100"
    exit 1
fi

SERVER_IP=$1
PORT=9092

echo "=========================================="
echo " Starting Federated Learning Client"
echo "=========================================="
echo "📍 Connecting to: $SERVER_IP:$PORT"
echo ""

# 서버 연결 테스트
echo "Testing connection..."
if ! nc -z $SERVER_IP $PORT 2>/dev/null; then
    echo "Cannot connect to server at $SERVER_IP:$PORT"
    echo "   Please check:"
    echo "   1. Server is running"
    echo "   2. Network connectivity"
    echo "   3. Firewall settings"
    exit 1
fi

echo "Server is reachable"
echo ""

# SuperNode 시작 (TCP Keepalive 설정 추가)
echo "🔗 Starting SuperNode with TCP Keepalive..."
# gRPC Keepalive 환경변수 설정
export GRPC_KEEPALIVE_TIME_MS=10000           # 10초마다 keepalive ping
export GRPC_KEEPALIVE_TIMEOUT_MS=5000         # 5초 응답 없으면 연결 끊김
export GRPC_CLIENT_KEEPALIVE_TIME_MS=10000    # 클라이언트 keepalive
export GRPC_CLIENT_KEEPALIVE_TIMEOUT_MS=5000  # 클라이언트 timeout

flower-supernode --insecure --superlink $SERVER_IP:$PORT

echo "=========================================="

