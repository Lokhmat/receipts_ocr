from database import get_db, Task
from json_parser import JsonParser

json_parser = JsonParser()

def process_task(task_id: str):
    db_gen = get_db()  # Get the generator
    db = next(db_gen)  # Retrieve the session
    try:
        # Get the task
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            print(f"Task {task_id} not found")
            return

        try:
            # Parse image directly to JSON
            json_data = json_parser.parse_image(task.image_data)
            
            # Update task with JSON result
            task.extracted_json = json_data

            db.commit()
            print(f"Task {task_id} processed successfully")
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            db.rollback()
    finally:
        # Clean up the session by exhausting the generator
        try:
            next(db_gen)
        except StopIteration:
            pass