# Security policy

Laika is under active development. The initial FastAPI task service has upload and resource controls, but it is not a multi-user production service. Please report security issues privately to the repository owner rather than posting image samples, secrets, or exploitable details in a public issue. A dedicated security contact is still needed before production deployment.

Implemented API controls include decoded JPEG/PNG/WebP validation, animated-image rejection, upload-byte and pixel limits, an overall multipart-body limit, UUID-isolated task storage, bounded queue/workers, disk budget checks, per-task random capability tokens, TTL cleanup, and deletion of failed/cancelled task data. The service binds to `127.0.0.1` by default. Task state is in-memory and the service currently runs as one Uvicorn process.

The access token is not an account identity or user authentication system. Before exposing the service to a network, deploy it behind an authenticated HTTPS boundary and apply suitable request rate limits. The API stores task images temporarily and returns temporary source, preview, and result links authorized by that task token. See [API deployment notes](docs/API.md).
