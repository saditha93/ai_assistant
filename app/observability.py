import logging
import sys

import structlog

from app.config import settings


def setup_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=settings.log_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
    )


log = structlog.get_logger()


def send_langsmith_feedback(run_id: str, score: int, comment: str | None) -> bool:
    """Attach a thumbs up/down to the trace of one chat turn. Returns False when LangSmith is off."""
    if not settings.has_langsmith:
        return False
    from langsmith import Client

    try:
        Client().create_feedback(run_id, key="user_rating", score=score, comment=comment, trace_id=run_id)
        return True
    except Exception as exc:
        log.warning("langsmith_feedback_failed", error=str(exc))
        return False
