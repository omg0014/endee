#!/bin/bash

# This script helps maintain a stable connection to Streamlit Cloud.
# We use a subdomain so your URL doesn't change every time you restart.

SUBDOMAIN="omg-endee-$(date +%s | cut -c 7-10)"

echo "-------------------------------------------------------"
echo "🚀 Starting Localtunnel on Port 8080..."
echo "🔗 Your stable URL will be: https://$SUBDOMAIN.loca.lt"
echo "-------------------------------------------------------"
echo "Note: If this subdomain is taken, localtunnel will give you a random one."
echo "Keep this terminal open while using the Cloud App!"
echo ""

npx localtunnel --port 8080 --subdomain $SUBDOMAIN
