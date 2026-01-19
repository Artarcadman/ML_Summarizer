from pydantic import BaseModel, Field

# Контракт запроса INPUT
class SummaryRequest(BaseModel):
    """
    Схема Pydantic для входных данных запроса суммаризации.
    Определяет ожидаемое поле ввода.
    Если формат запроса не соответсвует требованиям, он не пойдет в модель.
    """
    text: str = Field(
        description="Текст, который необходимо суммировать. Минимум 50 символов."
    )


# Контракт Ответа OUTPUT

class SummaryResponse(BaseModel):
    """
    Схема Pydantic для выходных данных ответа суммаризации.
    """
    original_text_length: int = Field(
        description="Длина оригинального текста."
    )
    summary: str = Field(
        description="Суммаризированный текст, сгенерированный ML-моделью."
    )
    summarized_text_length: int = Field(
        description="Длина сгенерированного суммаризированного текста."
    )
    model_name: str = Field(
        description="Имя модели, которая выполнила суммаризацию."
    )