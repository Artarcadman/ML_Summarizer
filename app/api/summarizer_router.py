from fastapi import APIRouter, Depends, HTTPException
from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services.summarizer import get_summarizer_service, SummarizerService

# Создаем APIRouter для группировки связанных эндпоинтов
router = APIRouter()

# FastAPI гарантирует, что мы всегда получаем один и тот же,
# уже инициализированный SummarizerService.

@router.post("/summarize", response_model=SummaryResponse)
def generate_summary(
    request_data: SummaryRequest,
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Принимает текст и возвращает его суммаризацию.
    """
    try:
        # 1. Вызов бизнес-логики (ML-сервиса)
        # Мы вызываем метод summarize, который находится в другом модуле
        # и изолирован от деталей HTTP.
        summary_text = summarizer.summarize(request_data.text)

        # 2. Формирование ответа согласно Pydantic-схеме SummaryResponse
        return SummaryResponse(
            original_text_length=len(request_data.text),
            summary=summary_text,
            model_name=summarizer.MODEL_NAME
        )
    except RuntimeError as e:
        # Если сервис не был инициализирован, вернем 500 ошибку
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Обработка других неожиданных ошибок
        print(f"Непредвиденная ошибка при суммаризации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке ML-запроса.")