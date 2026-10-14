from app.api.learning._shared import router_for_paths

router = router_for_paths(
    {
        "/study/sources",
        "/study/sources/{source_id}/open",
        "/study/sources/{source_id}/slides/{slide_number}",
    }
)
