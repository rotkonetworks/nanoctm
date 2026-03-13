#!/bin/sh
# Start Rust API in background
TICKS_FILE=/usr/share/nginx/html/ticks.json PORT=8081 ctm-api &

# Start nginx in foreground
exec nginx -g 'daemon off;'
