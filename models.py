from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String(255), nullable=False)
    department = Column(String(255), nullable=True)
    description = Column(Text, nullable=False)
    requirements = Column(Text, nullable=True)
    posted_date = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)

    applications = relationship("Application", back_populates="job")

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    resume_filename = Column(String(255), nullable=False)
    relevance_score = Column(Float, nullable=False)
    verdict = Column(String(50))
    missing_skills = Column(Text, nullable=True)
    feedback = Column(Text)
    application_date = Column(DateTime, default=datetime.datetime.utcnow)

    job = relationship("Job", back_populates="applications")