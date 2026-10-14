"""Learning API organized by student workflow."""

from app.api.learning.assessments import router as assessments_router
from app.api.learning.courses import router as courses_router
from app.api.learning.dashboard import router as dashboard_router
from app.api.learning.sources import router as sources_router
from app.api.learning.tutor import router as tutor_router

__all__ = [
    "assessments_router",
    "courses_router",
    "dashboard_router",
    "sources_router",
    "tutor_router",
]
