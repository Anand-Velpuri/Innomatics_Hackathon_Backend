from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base, Job, Application
from schemas import JobCreate, ApplicationCreate
from typing import Optional
import json
from sqlalchemy import func
import os


# Format: "mysql+mysqlclient://<user>:<password>@<host>/<dbname>"

DATABASE_URL=os.environ.get("DATABASE_URL")


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_database_tables():
    Base.metadata.create_all(bind=engine)

# --- Job CRUD functions ---
def create_job(db: Session, job: JobCreate):
    db_job = Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

def get_job(db: Session, job_id: int):
    return db.query(Job).filter(Job.id == job_id).first()

def get_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Job).filter(Job.is_active == True).offset(skip).limit(limit).all()

# --- Application CRUD function ---
def create_application(db: Session, application: ApplicationCreate):
    # Convert the python object to a dictionary
    app_data = application.model_dump()
    
    # Convert the list of skills to a JSON string for DB storage
    if app_data['missing_skills'] is not None:
        app_data['missing_skills'] = json.dumps(app_data['missing_skills'])
        
    db_application = Application(**app_data)
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_applications(db: Session, skip: int = 0, limit: int = 100):
    # Order by date to show the most recent applications first
    return db.query(Application).order_by(Application.application_date.desc()).offset(skip).limit(limit).all()


def count_total_applications(db: Session) -> int:
    return db.query(Application).count()

def calculate_overall_avg_score(db: Session) -> Optional[float]:
    # The scalar() method returns the first element of the first result or None
    return db.query(func.avg(Application.relevance_score)).scalar()

def count_open_positions(db: Session) -> int:
    return db.query(Job).filter(Job.is_active == True).count()

def count_high_fit_candidates(db: Session) -> int:
    return db.query(Application).filter(Application.verdict == 'High Suitability').count()