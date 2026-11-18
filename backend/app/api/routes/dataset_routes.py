from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])
dataset_service = DatasetService()


class SelectExistingRequest(BaseModel):
    file_name: str


@router.get("/list")
def list_datasets():
    """Returns list of all dataset files available on the server in uploads/."""
    try:
        files = dataset_service.list_uploaded_files()
        return {"success": True, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-existing")
async def select_existing_dataset(req: SelectExistingRequest):
    """Loads and processes an existing dataset from uploads/ without needing to re-upload."""
    try:
        result = await run_in_threadpool(dataset_service.process_existing_file, req.file_name)
        return {
            "success": True,
            "data": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Execute in threadpool so heavy dataframe processing never blocks FastAPI's event loop
        result = await run_in_threadpool(dataset_service.process_dataset, file)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")
