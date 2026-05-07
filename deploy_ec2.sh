set -e


REPO_URL="https://github.com/Aniroodh1234/sales_forcast_AI.git"
PROJECT_DIR="sales_forcast_AI"
S3_BUCKET="s3://sales-615645510621"

echo "==========================================================="
echo "   SalesCast AI - Automated Production Deployment Script"
echo "==========================================================="

echo -e "\n>>> 1. Updating System Packages..."
sudo apt update && sudo apt upgrade -y

echo -e "\n>>> 2. Installing System Dependencies..."
sudo apt install -y awscli git redis-server mysql-server nginx python3-pip python3-venv curl build-essential libmysqlclient-dev

echo -e "\n>>> 3. Installing Node.js (v22.x)..."
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

echo -e "\n>>> 4. Cloning Repository..."
cd ~
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory $PROJECT_DIR already exists. Pulling latest changes..."
    cd $PROJECT_DIR
    git pull
else
    git clone $REPO_URL $PROJECT_DIR
    cd $PROJECT_DIR
fi

echo -e "\n>>> 5. S3 Artifact Synchronization..."
echo "Checking AWS CLI configuration..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "AWS Credentials not found. Please enter them now to access S3 (or attach an IAM role to this EC2 instance)."
    aws configure
fi

echo "Syncing ML models and artifacts from $S3_BUCKET..."
mkdir -p artifacts

aws s3 sync "$S3_BUCKET" ./artifacts/ --exclude "*" \
    --include "preprocessor/*" \
    --include "models/*" \
    --include "metrics/*" \
    --include "data/*"

echo -e "\n>>> 6. Configuring MySQL Database..."
sudo mysql -e "CREATE DATABASE IF NOT EXISTS salescast_db;"
sudo mysql -e "CREATE USER IF NOT EXISTS 'salescast_user'@'localhost' IDENTIFIED BY 'secure_password_here';"
sudo mysql -e "GRANT ALL PRIVILEGES ON salescast_db.* TO 'salescast_user'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"
if [ -f "deployment/sql/init_schema.sql" ]; then
    sudo mysql salescast_db < deployment/sql/init_schema.sql || echo "Schema init failed. Continuing..."
fi

echo -e "\n>>> 7. Starting Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

echo -e "\n>>> 8. Setting up Python Backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
touch .env
cd ..

echo -e "\n>>> 9. Building Next.js Frontend..."
cd frontend
npm install
npm run build
cd ..

echo -e "\n>>> 10. Configuring NGINX Reverse Proxy..."
sudo cp deployment/nginx/salesforecast.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/salesforecast.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo -e "\n>>> 11. Configuring Systemd Services..."
CURRENT_DIR=$(pwd)

sed -i "s|/home/ubuntu/Sales_forcasting_System|$CURRENT_DIR|g" deployment/systemd/salesforecast-backend.service
sed -i "s|/home/ubuntu/Sales_forcasting_System|$CURRENT_DIR|g" deployment/systemd/salesforecast-frontend.service

sudo cp deployment/systemd/salesforecast-backend.service /etc/systemd/system/
sudo cp deployment/systemd/salesforecast-frontend.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable salesforecast-backend
sudo systemctl enable salesforecast-frontend

sudo systemctl restart salesforecast-backend
sudo systemctl restart salesforecast-frontend

echo -e "\n==========================================================="
echo "   DEPLOYMENT COMPLETE!"
echo "   Your application is now live."
echo "   Access it by entering this EC2 instance's Public IP in your browser."
echo "==========================================================="
