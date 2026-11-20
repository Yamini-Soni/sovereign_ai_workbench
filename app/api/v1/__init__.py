
from fastapi import APIRouter

from app.api.v1.system import router as system_router
from app.api.v1.documents import router as documents_router
from app.api.v1.qa import router as qa_router
from app.api.v1.agent import router as agent_router
from app.api.v1.vision import router as vision_router


api_router = APIRouter()

api_router.include_router(system_router)
api_router.include_router(documents_router)
api_router.include_router(qa_router)
api_router.include_router(agent_router)
api_router.include_router(vision_router)
