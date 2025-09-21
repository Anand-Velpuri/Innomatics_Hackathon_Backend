import json
from pydantic import BaseModel, Field, field_validator
import datetime
from typing import Optional, List

# --- Job Schemas ---
class JobBase(BaseModel):
    job_title: str
    department: Optional[str] = None
    description: str
    requirements: Optional[str] = None

class JobCreate(JobBase):
    pass

class Job(JobBase):
    id: int
    posted_date: datetime.datetime
    is_active: bool

    class Config:
        from_attributes = True

# Schema for the pre-fill response from a document
class JobPreFillResponse(BaseModel):
    job_title: Optional[str] = None
    department: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None


class JobWithStats(Job):
    no_applicants: int
    avg_score: Optional[float] = None



# --- Application Schemas (Updated from AnalysisResult) ---
class ApplicationBase(BaseModel):
    relevance_score: float
    verdict: str
    missing_skills: Optional[List[str]] = None
    feedback: Optional[str] = None

class ApplicationCreate(ApplicationBase):
    job_id: int
    resume_filename: str

class Application(ApplicationBase):
    id: int
    job_id: int
    resume_filename: str
    application_date: datetime.datetime

    class Config:
        from_attributes = True

    @field_validator("missing_skills", mode='before')
    @classmethod
    def parse_missing_skills(cls, value):
        # This function runs before validation.
        # If the value from the DB is a string, it converts it to a list.
        if isinstance(value, str):
            return json.loads(value)
        return value


class JobInApplication(BaseModel):
    id: int
    job_title: str

    class Config:
        from_attributes = True

# A detailed Application schema that includes the nested job info
class ApplicationDetails(Application):
    job: JobInApplication


class DashboardMetrics(BaseModel):
    total_applications: int
    avg_score: Optional[float] = None
    open_positions: int
    high_fit_candidates: int