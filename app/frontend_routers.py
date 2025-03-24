from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import base64
import json

from database import get_db, Task
from background_tasks import process_task

router = APIRouter()

@router.post("/create_task")
async def create_task(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    contents = await file.read()
    
    # Create a new task
    task_id = Task.generate_task_id()
    new_task = Task(task_id=task_id, image_data=contents)
    
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    
    # Start background processing
    if background_tasks:
        background_tasks.add_task(process_task, task_id)
    
    return {"task_id": task_id}

@router.get("/get_task/{task_id}")
async def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    response = {"task_id": task.task_id}
    
    if task.extracted_json is None:
        response["status"] = "processing"
    else:
        response["status"] = "completed"
        if task.extracted_json:
            response["extracted_json"] = task.extracted_json
    
    return response
