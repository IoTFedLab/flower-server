@echo off
REM Windows 클라이언트 시작 스크립트

if "%1"=="" (
    echo Usage: start_client.bat ^<SERVER_IP^>
    echo Example: start_client.bat 192.168.1.100
    exit /b 1
)

set SERVER_IP=%1
set PORT=9092

echo ==========================================
echo  Starting Federated Learning Client
echo ==========================================
echo 📍 Connecting to: %SERVER_IP%:%PORT%
echo.

REM 서버 연결 테스트
echo Testing connection...
ping -n 1 %SERVER_IP% >nul 2>&1
if errorlevel 1 (
    echo ❌ Cannot reach server at %SERVER_IP%
    echo    Please check:
    echo    1. Server is running
    echo    2. Network connectivity
    echo    3. Firewall settings
    exit /b 1
)

echo ✅ Server is reachable
echo.

REM gRPC Keepalive 환경변수 설정
echo 🔗 Starting SuperNode with TCP Keepalive...
set GRPC_KEEPALIVE_TIME_MS=10000
set GRPC_KEEPALIVE_TIMEOUT_MS=5000
set GRPC_CLIENT_KEEPALIVE_TIME_MS=10000
set GRPC_CLIENT_KEEPALIVE_TIMEOUT_MS=5000

REM SuperNode 시작
flwr-supernode --insecure --superlink %SERVER_IP%:%PORT%

echo ==========================================

