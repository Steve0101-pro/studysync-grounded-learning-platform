from app.api.learning._shared import router_for_paths

router = router_for_paths(
    {
        "/study/quizzes",
        "/study/quizzes/{question_id}/answer",
        "/study/mock-exams",
        "/study/mock-exams/{exam_id}",
        "/study/mock-exams/{exam_id}/submit",
    }
)
