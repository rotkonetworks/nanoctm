FROM nginx:alpine

# Static site + WASM debugger
COPY index.html research.html invest.html /usr/share/nginx/html/
COPY ctm-debugger.js ctm-debugger_bg.wasm /usr/share/nginx/html/

# SolidJS dashboard (fallback when WebGL unavailable)
COPY ctm-app/dist/ /usr/share/nginx/html/ctm-app/

# Rust API binary (static musl, no runtime deps)
COPY ctm-api /usr/local/bin/ctm-api
RUN chmod +x /usr/local/bin/ctm-api

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Entrypoint: start API in background, nginx in foreground
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

CMD ["/entrypoint.sh"]
