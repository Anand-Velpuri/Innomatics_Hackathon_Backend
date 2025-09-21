from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
import db_helper
import schemas
import analysis_engine
from typing import List
from schemas import JobWithStats

app = FastAPI(title="Resume Analyzer and Job Board API")

# Create database tables on startup
@app.on_event("startup")
def on_startup():
    db_helper.create_database_tables()

# === RECRUITER ENDPOINTS ===

@app.post("/jobs/parse-document", response_model=schemas.JobPreFillResponse, tags=["Recruiter"])
async def parse_job_document(job_doc: UploadFile = File(...)):
    """
    Upload a job description (PDF/DOCX) to automatically extract details
    and pre-fill the job creation form.
    """
    try:
        doc_bytes = await job_doc.read()
        if job_doc.filename.endswith('.pdf'):
            doc_text = analysis_engine.extract_text_from_pdf(doc_bytes)
        else:
            doc_text = analysis_engine.extract_text_from_docx(doc_bytes)
        
        parsed_data = analysis_engine.parse_job_document_to_fill_form(doc_text)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {e}")

@app.post("/jobs/", response_model=schemas.Job, tags=["Recruiter"])
def create_new_job(job: schemas.JobCreate, db: Session = Depends(db_helper.get_db)):
    """
    Create and store a new job posting in the database.
    """
    return db_helper.create_job(db=db, job=job)


@app.get("/metrics/", response_model=schemas.DashboardMetrics, tags=["Recruiter"])
def get_dashboard_metrics(db: Session = Depends(db_helper.get_db)):
    """
    Get key metrics for the recruiter dashboard, including total applications,
    average score, open positions, and high-fit candidates.
    """
    total_apps = db_helper.count_total_applications(db)
    avg_score_val = db_helper.calculate_overall_avg_score(db)
    open_pos = db_helper.count_open_positions(db)
    high_fit = db_helper.count_high_fit_candidates(db)

    return {
        "total_applications": total_apps,
        "avg_score": round(avg_score_val, 2) if avg_score_val is not None else None,
        "open_positions": open_pos,
        "high_fit_candidates": high_fit,
    }

# === CANDIDATE ENDPOINTS ===

@app.get("/jobs/", response_model=List[JobWithStats], tags=["Candidate"])
def get_available_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(db_helper.get_db)):
    """
    Get a list of all active job postings, including the number of applicants
    and their average score.
    """
    jobs_from_db = db_helper.get_jobs(db, skip=skip, limit=limit)
    
    results = []
    for job in jobs_from_db:
        # Calculate the number of applicants
        no_applicants = len(job.applications)
        
        # Calculate the average score, handling the case of zero applicants
        if no_applicants > 0:
            total_score = sum(app.relevance_score for app in job.applications)
            avg_score = round(total_score / no_applicants, 2)
        else:
            avg_score = None

        # Combine the original job data with the new stats
        job_data = schemas.Job.from_orm(job).model_dump()
        job_data['no_applicants'] = no_applicants
        job_data['avg_score'] = avg_score
        
        results.append(job_data)
    
    return results

@app.post("/jobs/{job_id}/apply", response_model=schemas.Application, tags=["Candidate"])
async def apply_for_job(
    job_id: int,
    resume_file: UploadFile = File(...),
    db: Session = Depends(db_helper.get_db)
):
    """
    Apply for a specific job by uploading a resume.
    Returns the relevance score and feedback.
    """
    # 1. Verify the job exists
    job = db_helper.get_job(db, job_id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 2. Extract text from resume
    resume_bytes = await resume_file.read()
    if resume_file.filename.endswith('.pdf'):
        resume_text = analysis_engine.extract_text_from_pdf(resume_bytes)
    else:
        resume_text = analysis_engine.extract_text_from_docx(resume_bytes)

    # 3. Combine job details for analysis
    jd_text = f"Job Title: {job.job_title}\n\nDescription: {job.description}\n\nRequirements: {job.requirements}"
    
    # 4. Run the full analysis pipeline (reusing our existing logic)
    parsed_jd_skills = analysis_engine.parse_job_description(jd_text)
    hard_match = analysis_engine.calculate_hard_match(resume_text, parsed_jd_skills)
    semantic_match = analysis_engine.calculate_semantic_match(resume_text, jd_text)
    
    final_result = analysis_engine.get_final_verdict_and_feedback(
        hard_match, semantic_match, parsed_jd_skills
    )

    # 5. Create and save the application record
    application_data = schemas.ApplicationCreate(
        job_id=job_id,
        resume_filename=resume_file.filename,
        relevance_score=final_result['relevance_score'],
        verdict=final_result['verdict'],
        missing_skills=final_result['missing_skills'], 
        feedback=final_result['feedback']
    )
    return db_helper.create_application(db=db, application=application_data)


@app.post("/jobs/parse-document", response_model=List[schemas.JobPreFillResponse], tags=["Recruiter"])
async def parse_job_document(job_doc: UploadFile = File(...)):
    """
    Upload a job description (PDF/DOCX) to automatically extract details
    for one or more jobs found in the document.
    """
    try:
        doc_bytes = await job_doc.read()
        if job_doc.filename.endswith('.pdf'):
            doc_text = analysis_engine.extract_text_from_pdf(doc_bytes)
        else:
            doc_text = analysis_engine.extract_text_from_docx(doc_bytes)
        
        # The function now returns a dictionary {"jobs": [...]}, so we extract the list.
        parsed_data = analysis_engine.parse_job_document_to_fill_form(doc_text)
        return parsed_data.get("jobs", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {e}")


@app.get("/applications/", response_model=List[schemas.ApplicationDetails], tags=["Recruiter"])
def get_all_applications(skip: int = 0, limit: int = 100, db: Session = Depends(db_helper.get_db)):
    """
    Get a list of all candidate applications with their scores and job details.
    """
    applications = db_helper.get_applications(db, skip=skip, limit=limit)
    return applications

@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Analyzer API"}