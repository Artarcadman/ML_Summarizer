from fastapi import FastAPI
from app.api.summarizer_router import router as summarizer_router

app = FastAPI(
    title="ML Text Summarization Service MVP",
    version="0.1.0",
    description="REST API для суммаризации текста с использованием ML-модели (t5-small base)."
)


app.include_router(summarizer_router, prefix="/api/v1", tags=["Summarization"])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Summarizer Service is running."}