import logging

import app.logger  # noqa: F401

logger = logging.getLogger(__name__)


class LiveFeed:
    """Stub — live websocket feed. To be implemented in Phase 3."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        logger.info("LiveFeed stub initialised for %d symbols (Phase 3 implementation pending)", len(symbols))

    def start(self) -> None:
        logger.info("LiveFeed.start() called — not yet implemented (Phase 3)")

    def stop(self) -> None:
        logger.info("LiveFeed.stop() called — not yet implemented (Phase 3)")
