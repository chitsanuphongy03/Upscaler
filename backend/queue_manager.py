"""Job queue for upscaling tasks."""

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

DB_FILE = Path("jobs.json")


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Upscaling job data"""
    id: str
    input_path: str
    original_filename: str
    scale: int
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0
    target_resolution: Optional[int] = None
    upscale_method: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "input_path": self.input_path,
            "original_filename": self.original_filename,
            "scale": self.scale,
            "target_resolution": self.target_resolution,
            "upscale_method": self.upscale_method,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error": self.error,
            "progress": self.progress
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create Job from dictionary"""
        return cls(
            id=data["id"],
            input_path=data["input_path"],
            original_filename=data["original_filename"],
            scale=data["scale"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            output_path=data.get("output_path"),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            target_resolution=data.get("target_resolution"),
            upscale_method=data.get("upscale_method")
        )


class JobQueue:
    """Queue manager for upscaling jobs"""
    
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._pending_queue: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load jobs from database file"""
        if not DB_FILE.exists():
            return
            
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for job_data in data:
                try:
                    job = Job.from_dict(job_data)
                    self._jobs[job.id] = job
                    
                    if job.status == JobStatus.PROCESSING:
                        job.status = JobStatus.FAILED
                        job.error = "Interrupted by server restart"
                    elif job.status == JobStatus.PENDING:
                        self._pending_queue.append(job.id)
                        
                except Exception as e:
                    print(f"Error loading job: {e}")
                    
        except Exception as e:
            print(f"Failed to load jobs database: {e}")

    def save_jobs(self) -> None:
        """Save jobs to database file"""
        try:
            data = [job.to_dict() for job in self._jobs.values()]
            with open(DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save jobs database: {e}")
    
    async def add_job(self, job: Job) -> None:
        """Add job to queue"""
        async with self._lock:
            self._jobs[job.id] = job
            self._pending_queue.append(job.id)
            self.save_jobs()
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    async def get_all_jobs(self) -> list[Job]:
        """Get all jobs"""
        return list(self._jobs.values())
    
    async def get_next_job(self) -> Optional[Job]:
        """Get next pending job from queue"""
        async with self._lock:
            while self._pending_queue:
                job_id = self._pending_queue[0]
                job = self._jobs.get(job_id)
                
                if job and job.status == JobStatus.PENDING:
                    self._pending_queue.popleft()
                    return job
                else:
                    self._pending_queue.popleft()
        
        return None
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete job and its output file"""
        async with self._lock:
            job = self._jobs.get(job_id)
            
            if not job:
                return False
            
            if job.status == JobStatus.PENDING:
                try:
                    self._pending_queue.remove(job_id)
                except ValueError:
                    pass
            
            if job.output_path:
                try:
                    path = Path(job.output_path)
                    if path.exists():
                        path.unlink()
                        print(f"Deleted file: {path}")
                except Exception as e:
                    print(f"Error deleting file {job.output_path}: {e}")
            
            del self._jobs[job_id]
            self.save_jobs()
            
            return True
    
    async def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress"""
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress
    
    async def get_queue_position(self, job_id: str) -> int:
        """Get job position in queue (1-indexed, 0 if not in queue)"""
        try:
            return list(self._pending_queue).index(job_id) + 1
        except ValueError:
            return 0
    
    async def get_pending_count(self) -> int:
        """Get count of pending jobs"""
        count = 0
        for job_id in self._pending_queue:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                count += 1
        return count
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than specified hours"""
        async with self._lock:
            now = datetime.now()
            to_remove = []
            
            for job_id, job in self._jobs.items():
                if job.completed_at:
                    age_hours = (now - job.completed_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
            
            if to_remove:
                self.save_jobs()
            
            return len(to_remove)
