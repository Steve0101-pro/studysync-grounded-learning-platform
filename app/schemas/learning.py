from pydantic import BaseModel


class LearningCourseSummary(BaseModel):
    id: str
    title: str
    source_count: int
    passage_count: int
