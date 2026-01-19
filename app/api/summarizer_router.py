from fastapi import APIRouter, Depends, HTTPException
from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services.summarizer import get_summarizer_service, SummarizerService

# Создаем APIRouter для группировки связанных эндпоинтов
router = APIRouter()

@router.post("/summarize", response_model=SummaryResponse)
def generate_summary(
    request_data: SummaryRequest,
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Принимает текст и возвращает его суммаризацию через ML-модель T5.
    """
    try:
        # 1. Вызов бизнес-логики (ML-сервиса)
        # Мы вызываем метод summarize, который использует веса из RAM
        summary_text = summarizer.summarize(request_data.text)

        # 2. Формирование ответа согласно Pydantic-схеме SummaryResponse
        # Теперь берем имя модели из self.model_name экземпляра
        return SummaryResponse(
            original_text_length=len(request_data.text),
            summary=summary_text,
            summarized_text_length = len(summary_text),
            model_name=summarizer.model_name
        )
        
    except RuntimeError as e:
        # Ошибка инициализации (например, если папка ml_model не найдена)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Обработка непредвиденных ошибок во время инференса
        print(f"Непредвиденная ошибка при суммаризации: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Внутренняя ошибка сервера при обработке ML-запроса."
        )