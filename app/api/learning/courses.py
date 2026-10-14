from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models import StudyContentUnit, StudySource, User
from app.schemas.learning import LearningCourseSummary

router = APIRouter(prefix="/study", tags=["Learning courses"])


@router.get("/courses", response_model=list[LearningCourseSummary])
async def list_learning_courses(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    source_count = await db.scalar(
        select(func.count(StudySource.id)).where(
            StudySource.user_id == current_user.id
        )
    )
    passage_count = await db.scalar(
        select(func.count(StudyContentUnit.id))
        .join(StudySource, StudySource.id == StudyContentUnit.source_id)
        .where(StudySource.user_id == current_user.id)
    )
    if not source_count:
        return []
    return [
        {
            "id": "my-study-library",
            "title": "My study library",
            "source_count": source_count,
            "passage_count": passage_count or 0,
        }
    ]
