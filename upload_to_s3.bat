@echo off
echo =======================================================
echo    SalesCast AI - S3 Artifact Upload Script
echo =======================================================

set S3_BUCKET=s3://sales-615645510621/artifacts/

echo Checking AWS CLI installation...
where aws >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] AWS CLI is not installed or not in your PATH.
    echo Please install it from: https://aws.amazon.com/cli/
    exit /b 1
)

echo Syncing local 'artifacts' directory to %S3_BUCKET%...
aws s3 sync .\artifacts\ %S3_BUCKET%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] All models, metrics, and data have been uploaded to S3!
) else (
    echo.
    echo [ERROR] Failed to upload to S3. Please ensure you have run 'aws configure' and have the correct permissions.
)
pause
