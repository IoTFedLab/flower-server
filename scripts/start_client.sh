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

# SuperNode 시작
echo "🔗 Starting SuperNode..."
flwr-supernode --insecure --superlink $SERVER_IP:$PORT

echo "=========================================="

