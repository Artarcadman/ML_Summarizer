from transformers import pipeline
import os


GLOBAL_MODEL_NAME = "Artemkaaa/t5-small-summarizer-xlsum"


_summarizer_instance = None
_summarizer_pipeline = None


class SummarizerService:
    
    def __init__(self):
        self.model_name = GLOBAL_MODEL_NAME

    def initialize(self):
        global _summarizer_pipeline
        
        if _summarizer_pipeline is not None:
            return

        print(f"Загрузка модели из репозитория: {self.model_name}")
        
        _summarizer_pipeline = pipeline(
            "summarization", 
            model=self.model_name, 
            device="cpu"
        )
        
        print("Модель активна!")
    

    def summarize(self, text: str) -> str:
        if _summarizer_pipeline is None:
            raise RuntimeError("Модель не загружена.")

        print("Выполнение инференса...")
        
        # Параметры для улучшения качества
        result = _summarizer_pipeline(
            text, 
            max_length=150, 
            min_length=30,
            num_beams=2,
            early_stopping=True
        )
        
        print("Инференс завершен.")
        return result[0]['summary_text']

def get_summarizer_service() -> SummarizerService:
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = SummarizerService()
    _summarizer_instance.initialize() 
    return _summarizer_instance