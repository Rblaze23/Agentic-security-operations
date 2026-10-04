"""`secops-api`: run the detection API with uvicorn."""

from __future__ import annotations

import uvicorn

from secops.api.settings import get_api_settings


def main() -> None:
    settings = get_api_settings()
    uvicorn.run(
        "secops.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
