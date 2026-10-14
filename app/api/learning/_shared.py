from fastapi import APIRouter
from fastapi.routing import APIRoute

from app.api.study import router as legacy_study_router


def router_for_paths(paths: set[str]) -> APIRouter:
    router = APIRouter()
    for route in legacy_study_router.routes:
        if isinstance(route, APIRoute) and route.path in paths:
            router.routes.append(route)
    return router
