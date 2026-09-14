# AI for DB

Services that make decisions for the database engine live here. This is separate
from `aiengine/runtime`, which executes analytics tasks over streamed data.

The implemented service is [query_opt](query_opt/README.md): the NQO action
server, policy models, experience storage, and explicit training commands.
It uses a small HTTP control protocol with the PostgreSQL backend, not the
analytics WebSocket batch-data protocol. No shared service framework or empty
future-service directories are introduced.
