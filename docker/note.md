If the `ERROR: timezone directory stack overflow` error pops up when trying to connect to the host:
1. Run `docker-compose exec tile-server bash`
2. Run `unlink /usr/share/zoneinfo/localtime`
---
