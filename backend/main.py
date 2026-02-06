"""
Upscaler Backend - FastAPI Server
AI-powered image and video upscaling with Real-ESRGAN
"""

import asyncio
import logging
import os
import re
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket

from queue_manager import Job, JobQueue, JobStatus
from upscaler import VideoUpscaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
VALID_RESOLUTIONS = [720, 1080, 1440, 2160]
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50 MB

for d in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
    d.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and remove dangerous characters.
    Only allows alphanumeric, underscore, hyphen, and dot.
    """
    if not filename:
        return "unnamed"
    
    # Get basename to prevent path traversal
    filename = os.path.basename(filename)
    
    # Remove null bytes and other dangerous characters
    filename = filename.replace('\x00', '')
    
    # Split name and extension
    name, ext = os.path.splitext(filename)
    
    # Only allow safe characters in filename (alphanumeric, underscore, hyphen, space)
    name = re.sub(r'[^\w\s\-]', '_', name, flags=re.UNICODE)
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscore
    name = name.strip('_')  # Remove leading/trailing underscores
    
    # Limit length
    if len(name) > 200:
        name = name[:200]
    
    # Ensure we have a name
    if not name:
        name = "file"
    
    # Sanitize extension (only allow known safe extensions)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ""
    
    return f"{name}{ext}" if ext else name


def validate_path(file_path: str, allowed_dirs: list[Path]) -> Path:
    """
    Validate that file path is within allowed directories.
    Prevents path traversal attacks.
    """
    try:
        # Resolve to absolute path
        resolved = Path(file_path).resolve()
        
        # Check if path is within any allowed directory
        for allowed_dir in allowed_dirs:
            allowed_resolved = allowed_dir.resolve()
            if str(resolved).startswith(str(allowed_resolved)):
                return resolved
        
        raise ValueError(f"Path not in allowed directories")
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")


def is_safe_path(file_path: str, base_dir: Path) -> bool:
    """Check if path is within base directory."""
    try:
        return str(Path(file_path).resolve()).startswith(str(base_dir.resolve()))
    except Exception:
        return False


class EndpointFilter(logging.Filter):
    """Filter out noisy endpoint logs (status, job polling)"""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "GET /api/status" in msg:
            return False
        if "GET /api/jobs/" in msg:
            return False
        return True


# -----------------------------------------------------------------------------
# App state
# -----------------------------------------------------------------------------
job_queue = JobQueue()
upscaler: Optional[VideoUpscaler] = None
active_websockets: dict[str, WebSocket] = {}
model_status = "offline"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global upscaler, model_status
    
    # Filter noisy logs
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
    
    model_status = "loading"
    print("🚀 Loading Real-ESRGAN model...")
    
    try:
        loop = asyncio.get_running_loop()
        upscaler = await loop.run_in_executor(None, VideoUpscaler)
        model_status = "ready"
        print("✅ Model loaded successfully!")
    except Exception as e:
        model_status = "failed"
        print(f"❌ Failed to load model: {e}")
    
    # Start background tasks
    asyncio.create_task(process_jobs())
    asyncio.create_task(cleanup_old_files())
    
    yield
    
    print("🛑 Shutting down server...")


app = FastAPI(
    title="Upscaler API",
    description="AI-powered image and video upscaling API (Real-ESRGAN)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
# - ในโหมด dev (รันจากโค้ดปกติ) ต้องเปิด CORS เพราะ frontend (Vite) กับ backend คนละพอร์ต
# - แต่ในโหมด .exe (PyInstaller, sys.frozen=True) frontend ถูกเสิร์ฟจาก backend host เดียวกันอยู่แล้ว
#   และ CORSMiddleware บางครั้งจะ block WebSocket ด้วย 403 → progress ใช้ไม่ได้
# เพราะงั้น:
#   - ถ้าไม่ใช่ .exe → เปิด CORS ปกติ
#   - ถ้าเป็น .exe → ปิด CORS ไปเลย ปลอดภัยเพราะรันแค่ local app
if not getattr(sys, "frozen", False):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Static file serving
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")


async def cleanup_old_files():
    """Remove files older than 1 hour from uploads and temp directories"""
    while True:
        try:
            now = time.time()
            max_age = 3600  # 1 hour
            
            for directory in [UPLOAD_DIR, TEMP_DIR]:
                if not directory.exists():
                    continue
                    
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_age = now - file_path.stat().st_mtime
                        if file_age > max_age:
                            try:
                                file_path.unlink()
                                print(f"🧹 Cleaned up: {file_path}")
                            except Exception as e:
                                print(f"❌ Cleanup error ({file_path}): {e}")
        except Exception as e:
            print(f"❌ Cleanup task error: {e}")
            
        await asyncio.sleep(900)  # Run every 15 minutes


async def process_jobs():
    """Background task to process upscaling jobs from queue"""
    global upscaler
    
    while True:
        job = await job_queue.get_next_job()
        
        if job and upscaler:
            try:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                job_queue.save_jobs()
                
                await notify_progress(job.id, {
                    "status": "processing",
                    "progress": 0,
                    "message": "Starting..."
                })

                last_logged_pct = [-1]

                async def on_progress(progress: float, message: str):
                    # อัปเดตลง job object สำหรับ /api/jobs (ใช้กับ polling)
                    await job_queue.update_progress(job.id, progress)
                    job_queue.save_jobs()

                    # Log แค่ % ทุก 10% และตอน 100%
                    p = int(progress)
                    if p >= 100:
                        print("Upscale: 100% เสร็จแล้ว")
                        last_logged_pct[0] = 100
                    elif last_logged_pct[0] < 0 or (p // 10) > (last_logged_pct[0] // 10):
                        print(f"Upscale: {p}%")
                        last_logged_pct[0] = p

                    # ส่งไปทาง WebSocket (ถ้ามี)
                    await notify_progress(job.id, {
                        "status": "processing",
                        "progress": progress,
                        "message": message
                    })
                
                # Determine file type and process
                file_ext = Path(job.input_path).suffix.lower()
                is_video = file_ext in VIDEO_EXTENSIONS
                
                if is_video:
                    output_path, method = await upscaler.upscale_video(
                        input_path=job.input_path,
                        output_dir=OUTPUT_DIR,
                        scale=job.scale,
                        progress_callback=on_progress
                    )
                else:
                    await notify_progress(job.id, {
                        "status": "processing",
                        "progress": 50,
                        "message": "Upscaling image..."
                    })
                    output_path, method = await upscaler.upscale_image_file(
                        input_path=job.input_path,
                        output_dir=OUTPUT_DIR,
                        scale=job.scale
                    )
                
                # Mark as completed
                job.output_path = str(output_path)
                job.upscale_method = method
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.completed_at = datetime.now()
                job_queue.save_jobs()
                
                await notify_progress(job.id, {
                    "status": "completed",
                    "progress": 100,
                    "message": "Done!",
                    "output_path": job.output_path,
                    "upscale_method": job.upscale_method
                })
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
                job_queue.save_jobs()
                
                await notify_progress(job.id, {
                    "status": "failed",
                    "progress": 0,
                    "message": f"Error: {str(e)}"
                })
        
        await asyncio.sleep(1)


async def notify_progress(job_id: str, data: dict):
    """Send progress update via WebSocket"""
    if job_id in active_websockets:
        try:
            await active_websockets[job_id].send_json(data)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Upscaler API ready (supports images and videos)"
    }


@app.get("/api/status")
async def get_model_status():
    """Get AI model status"""
    return {"status": model_status}


# Fallback for get_video_info when bundled .exe truncates this file
try:
    get_video_info  # type: ignore[name-defined]
except NameError:
    async def get_video_info(_path: Path) -> dict:  # type: ignore[no-redef]
        return {}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload image or video file"""
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    
    # Check file size (approximate using content-length header if available)
    # Note: This is soft check, we verify real size during read
    content_length = file.headers.get("content-length")
    limit = MAX_VIDEO_SIZE if file_ext in VIDEO_EXTENSIONS else MAX_IMAGE_SIZE
    
    if content_length and int(content_length) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Limit is {limit // (1024*1024)}MB"
        )
    
    file_type = "video" if file_ext in VIDEO_EXTENSIONS else "image"
    
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}" 
    file_path = UPLOAD_DIR / filename
    
    # Save file and check size
    size = 0
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds limit of {limit // (1024*1024)}MB"
                    )
                await f.write(chunk)
    except Exception as e:
        # Cleanup partial file
        if file_path.exists():
            file_path.unlink()
        raise e
    
    # Get media info
    media_info = {}
    if file_type == "video":
        media_info = await get_video_info(file_path)
    else:
        import cv2
        img = cv2.imread(str(file_path))
        if img is not None:
            h, w = img.shape[:2]
            media_info = {"width": w, "height": h}
    
    return {
        "file_id": file_id,
        "filename": safe_filename,
        "path": str(file_path),
        "size": size,
        "file_type": file_type,
        "media_info": media_info
    }


@app.post("/api/upscale-image")
async def upscale_image(
    file_path: str,
    scale: int = 2,
    original_filename: str = "image"
):
    """Upscale image immediately (synchronous)"""
    global upscaler
    
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    try:
        # Validate path
        input_path = validate_path(file_path, [UPLOAD_DIR, TEMP_DIR])
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        if not upscaler:
            raise HTTPException(status_code=500, detail="Upscaler not ready")
        
        # Sanitize original filename for output
        safe_original = sanitize_filename(original_filename)
        safe_original_stem = Path(safe_original).stem
    
        import cv2
        
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot read image file")
        
        upscaled, method = await upscaler.upscale_frame(img, scale=scale)
        
        # Format: Upscaled_[Scale]x_[Name].png
        output_filename = f"Upscaled_{scale}x_{safe_original_stem}.png"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), upscaled)
        
        return {
            "status": "completed",
            "output_path": str(output_path),
            "output_url": f"/outputs/{output_filename}",
            "original_size": {"width": img.shape[1], "height": img.shape[0]},
            "upscaled_size": {"width": upscaled.shape[1], "height": upscaled.shape[0]},
            "upscale_method": method
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/upscale")
async def start_upscale(
    file_path: str,
    target_resolution: int = 1080,
    original_filename: str = "video"
):
    """Queue video/image for upscaling"""
    if target_resolution not in VALID_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Resolution must be one of: {VALID_RESOLUTIONS}"
        )
    
    try:
        # Validate path prevents path traversal
        valid_path = validate_path(file_path, [UPLOAD_DIR])
        if not valid_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Sanitize filename
        safe_filename = sanitize_filename(original_filename)

        # Calculate scale factor
        input_height = 720
        file_ext = valid_path.suffix.lower()
        
        if file_ext in IMAGE_EXTENSIONS:
            import cv2
            img = cv2.imread(str(valid_path))
            if img is not None:
                input_height = img.shape[0]
        else:
            video_info = await get_video_info(valid_path)
            if video_info.get("height"):
                input_height = video_info["height"]
        
        scale = max(2, min(4, target_resolution // input_height))
        
        # Create job
        job = Job(
            id=str(uuid.uuid4()),
            input_path=str(valid_path),
            original_filename=safe_filename,
            scale=scale,
            target_resolution=target_resolution,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        
        await job_queue.add_job(job)
        
        return {
            "job_id": job.id,
            "status": job.status.value,
            "message": f"Job queued (upscale {scale}x)"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    jobs = await job_queue.get_all_jobs()
    return {"jobs": [job.to_dict() for job in jobs]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status"""
    job = await job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and output file"""
    success = await job_queue.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted"}


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download upscaled result"""
    job = await job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = Path(job.output_path) if job.output_path else None
    
    # Security check: Ensure output path is within OUTPUT_DIR
    if not output_path or not output_path.exists() or not is_safe_path(str(output_path), OUTPUT_DIR):
        raise HTTPException(status_code=404, detail="Output file not found or invalid path")
    
    # Determine media type
    ext = output_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        media_type = "image/png"
    elif ext in VIDEO_EXTENSIONS:
        media_type = "video/mp4"
    else:
        media_type = "application/octet-stream"
    
    # Generate download filename
    original_name = Path(sanitize_filename(job.original_filename)).stem if job.original_filename else "output"
    
    # Determine quality suffix
    if job.target_resolution:
        quality = f"{job.target_resolution}p"
    else:
        quality = f"{job.scale}x"
        
    # Format: Upscaled_[Quality]_[Name].[ext]
    download_filename = f"Upscaled_{quality}_{original_name}{ext}"
    
    return FileResponse(
        path=str(output_path),
        filename=download_filename,
        media_type=media_type
    )


@app.get("/api/preview/{file_id}")
async def get_preview_frame(file_id: str, time_sec: float = 1.0):
    """Generate preview with AI upscaling comparison"""
    global upscaler
    
    # Find file - Sanitize ID just in case (should be UUID)
    safe_file_id = sanitize_filename(file_id) # Though file_id is mostly UUID, sanitizing removes path traversal chars
    
    file_path = None
    for ext in [*VIDEO_EXTENSIONS, *IMAGE_EXTENSIONS]:
        path = UPLOAD_DIR / f"{safe_file_id}{ext}"
        if path.exists():
            file_path = path
            break
            
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Extra safety check
    if not is_safe_path(str(file_path), UPLOAD_DIR):
        raise HTTPException(status_code=403, detail="Invalid file path")
        
    is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
    
    if not upscaler:
        raise HTTPException(status_code=500, detail="Upscaler not ready")
    
    import cv2
    import numpy as np
    
    try:
        if is_video:
            original_frame = await upscaler.extract_frame(file_path, time_sec)
        else:
            stream = open(file_path, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            original_frame = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            
            if original_frame is not None and original_frame.shape[2] == 4:
                trans_mask = original_frame[:, :, 3] == 0
                original_frame[trans_mask] = [255, 255, 255, 255]
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
        
        if original_frame is None:
            raise ValueError("Could not decode image")

        # Upscale for preview
        upscaled_frame, method = await upscaler.upscale_frame(original_frame)
        
        # Save preview files
        orig_path = TEMP_DIR / f"{safe_file_id}_original.jpg"
        upsc_path = TEMP_DIR / f"{safe_file_id}_upscaled.jpg"
        
        cv2.imwrite(str(orig_path), original_frame)
        cv2.imwrite(str(upsc_path), upscaled_frame)
        
        return {
            "original": f"/temp/{safe_file_id}_original.jpg",
            "upscaled": f"/temp/{safe_file_id}_upscaled.jpg",
            "method": method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


# -----------------------------------------------------------------------------
# Serve frontend (SPA)
# -----------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    # Try multiple common PyInstaller locations
    base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    exe_dir = Path(sys.executable).parent
    
    possible_paths = [
        base_dir / "dist",
        exe_dir / "dist",
        exe_dir / "_internal" / "dist",
        base_dir / "_internal" / "dist",
    ]
    
    frontend_dist = None
    for p in possible_paths:
        if p.exists() and (p / "index.html").exists():
            frontend_dist = p
            break
    
    if not frontend_dist:
        print("❌ Could not find frontend 'dist' folder in any of these locations:")
        for p in possible_paths:
            print(f"  - {p}")
        frontend_dist = possible_paths[0] # Fallback for error message
else:
    frontend_dist = Path("../dist")

if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    print(f"🌐 Serving frontend from: {frontend_dist}")
    # Mount assets
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
    
    # Explicit route for root
    @app.get("/")
    async def serve_index():
        print("🏠 Serving index.html to root request")
        return FileResponse(frontend_dist / "index.html")
    
    # Catch-all route to serve index.html
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not found")
        
        potential_file = frontend_dist / full_path
        if potential_file.exists() and potential_file.is_file():
            print(f"📄 Serving file: {full_path}")
            return FileResponse(potential_file)
            
        print(f"🔍 Route {full_path} not found, falling back to index.html")
        return FileResponse(frontend_dist / "index.html")
else:
    print(f"⚠️ Frontend dist not found or index.html missing at {frontend_dist}")

if __name__ == "__main__":
    import threading
    import uvicorn
    import webbrowser

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    print("🚀 Starting Upscaler AI...")
    print("👉 Open your browser at: http://localhost:8000")
    def _open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=_open_browser, daemon=True).start()

    # Disable reload for production/exe
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates"""
    print(f"🔌 WebSocket connection attempt: {job_id} from {websocket.client}")
    try:
        await websocket.accept()
        print(f"✅ WebSocket accepted: {job_id}")
        active_websockets[job_id] = websocket
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"❌ WebSocket error ({job_id}): {str(e)}")
    finally:
        if job_id in active_websockets:
            del active_websockets[job_id]
        print(f"🔌 WebSocket closed: {job_id}")


async def get_video_info(file_path: Path) -> dict:
    """Get video metadata via ffprobe."""
    try:
        import json
        import subprocess

        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(file_path)
        ], capture_output=True, text=True)
        
        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {}
        )
        
        return {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "30/1")),
            "codec": video_stream.get("codec_name")
        }
    except Exception:
        return {}



