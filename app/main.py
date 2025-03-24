from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frontend_routers import router
from database import create_tables
import uvicorn

app = FastAPI(title="Receipt Scanner")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()

# Include the router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 