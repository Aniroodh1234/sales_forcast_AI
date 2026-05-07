#!/bin/bash
# SalesCast AI - EC2 Deployment Provisioning Script (Ubuntu 24.04 LTS)
set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing dependencies (Redis, MySQL, Nginx, Python, Node.js)..."
sudo apt install -y redis-server mysql-server nginx python3-pip python3-venv git curl build-essential libmysqlclient-dev

# Install Node.js (v22.x)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Secure MySQL and create database
echo "Configuring MySQL..."
sudo mysql -e "CREATE DATABASE IF NOT EXISTS salescast_db;"
sudo mysql -e "CREATE USER IF NOT EXISTS 'salescast_user'@'localhost' IDENTIFIED BY 'secure_password_here';"
sudo mysql -e "GRANT ALL PRIVILEGES ON salescast_db.* TO 'salescast_user'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"
sudo mysql salescast_db < deployment/sql/init_schema.sql

echo "Configuring Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

echo "Setting up Python virtual environment for backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Configuring NGINX..."
sudo cp deployment/nginx/salesforecast.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/salesforecast.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "Setting up systemd services..."
sudo cp deployment/systemd/salesforecast-backend.service /etc/systemd/system/
sudo cp deployment/systemd/salesforecast-frontend.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable salesforecast-backend
sudo systemctl enable salesforecast-frontend

sudo systemctl start salesforecast-backend
sudo systemctl start salesforecast-frontend

echo "Deployment setup complete!"
