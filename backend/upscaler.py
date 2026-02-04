"""
Video Upscaler - AI-powered upscaling with Real-ESRGAN
"""

import os
import uuid
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class VideoUpscaler:
    """AI-powered video and image upscaler using Real-ESRGAN"""
    
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.use_torch = False
        self.upscaler = None
        self.device = None
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Real-ESRGAN model"""
        try:
            import torch
            import sys
            import types
            from torchvision.transforms import functional as F
            
            # Patch: Fix BasicSR compatibility with TorchVision > 0.16
            try:
                import torchvision.transforms.functional_tensor
            except ImportError:
                fake_module = types.ModuleType("torchvision.transforms.functional_tensor")
                fake_module.rgb_to_grayscale = F.rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = fake_module

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            print("🚀 Loading Real-ESRGAN model...")
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   Device: {self.device}")
            
            # Anime model (faster, good quality)
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=6, 
                num_grow_ch=32, 
                scale=4
            )
            
            self.upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=str(self.device).startswith('cuda'),
                device=self.device,
            )
            
            self.use_torch = True
            print("✅ Real-ESRGAN initialized!")
            
        except Exception as e:
            print(f"⚠️ AI Model init failed: {e}")
            print("   Falling back to OpenCV resize")
            self.use_torch = False

    async def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> Tuple[np.ndarray, str]:
        """
        Upscale a single frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            scale: Upscale factor (2 or 4)
            
        Returns:
            Tuple of (upscaled image, method used)
        """
        result_img = None
        method = "OpenCV"
        
        # Try AI upscaling first
        if self.use_torch and self.upscaler:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run in thread to avoid blocking
                loop = asyncio.get_running_loop()
                output, _ = await loop.run_in_executor(
                    None,
                    lambda: self.upscaler.enhance(img_rgb, outscale=scale)
                )
                
                result_img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                method = "Real-ESRGAN"
            except Exception as e:
                print(f"❌ AI upscale error: {e}")
        
        # Fallback to OpenCV
        if result_img is None:
            print(f"🔄 Using OpenCV resize ({scale}x)")
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            result_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            method = "OpenCV"
            
        return result_img, method
    
    async def upscale_video(
        self,
        input_path: str,
        output_dir: Path,
        scale: int = 2,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Path, str]:
        """
        Upscale entire video
        
        Args:
            input_path: Path to input video
            output_dir: Directory for output
            scale: Upscale factor
            progress_callback: Async callback for progress updates
            
        Returns:
            Tuple of (output path, method used)
        """
        input_path = Path(input_path)
        job_id = str(uuid.uuid4())
        frames_dir = self.temp_dir / f"frames_{job_id}"
        upscaled_dir = self.temp_dir / f"upscaled_{job_id}"
        
        frames_dir.mkdir(exist_ok=True)
        upscaled_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Extract frames
            if progress_callback:
                await progress_callback(10, "Extracting frames...")
            
            await self._extract_all_frames(input_path, frames_dir)
            
            # Step 2: Upscale frames
            frame_files = sorted(list(frames_dir.glob("*.png")))
            total_frames = len(frame_files)
            
            if total_frames == 0:
                raise RuntimeError("No frames extracted")
            
            print(f"🎬 Processing {total_frames} frames...")
            
            for i, frame_file in enumerate(frame_files):
                if progress_callback:
                    prog = 20 + int((i / total_frames) * 70)
                    await progress_callback(prog, f"Upscaling frame {i+1}/{total_frames}...")
                
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                result, _ = await self.upscale_frame(img, scale)
                cv2.imwrite(str(upscaled_dir / frame_file.name), result)
            
            # Step 3: Reconstruct video
            if progress_callback:
                await progress_callback(90, "Reconstructing video...")
            
            output_path = output_dir / f"{input_path.stem}_upscaled_{scale}x.mp4"
            await self._reconstruct_video(input_path, upscaled_dir, output_path)
            
            method = "Real-ESRGAN" if self.use_torch else "OpenCV"
            return output_path, method
            
        except Exception as e:
            # Log error
            self._log_error(input_path, e)
            raise
            
        finally:
            # Cleanup temp directories
            self._cleanup_temp_dirs(frames_dir, upscaled_dir)

    async def upscale_image_file(
        self,
        input_path: str,
        output_dir: Path,
        scale: int = 4
    ) -> Tuple[Path, str]:
        """
        Upscale image file
        
        Args:
            input_path: Path to input image
            output_dir: Directory for output
            scale: Upscale factor
            
        Returns:
            Tuple of (output path, method used)
        """
        input_path = Path(input_path)
        
        # Read image (handle unicode paths)
        with open(input_path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Handle alpha channel
        if len(img.shape) == 3 and img.shape[2] == 4:
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        # Upscale
        result, method = await self.upscale_frame(img, scale)
        
        # Save output
        output_filename = f"upscaled_{scale}x_{input_path.stem}.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), result)
        
        return output_path, method

    async def _extract_all_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract all frames from video using FFmpeg"""
        print(f"🎬 Extracting frames from {video_path}")
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            str(output_dir / "frame_%06d.png")
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        count = len(list(output_dir.glob("*.png")))
        print(f"✅ Extracted {count} frames")
        
        if count == 0:
            raise RuntimeError("No frames extracted")
            
        return count

    async def _reconstruct_video(self, original_video: Path, frames_dir: Path, output_path: Path) -> None:
        """Reconstruct video from frames with original audio"""
        print("🎬 Reconstructing video...")
        
        # Get original FPS
        cap = cv2.VideoCapture(str(original_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if not fps or fps <= 0:
            fps = 30
        
        temp_video = output_path.parent / f"temp_{output_path.name}"
        
        # Combine frames
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(temp_video)
        ]
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg reconstruct failed: {result.stderr}")
        
        # Add audio from original
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", str(original_video),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(output_path)
        ]
        
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        )
        
        # Cleanup temp video with retries
        for attempt in range(3):
            try:
                await asyncio.sleep(0.5)  # Wait for file handles to be released
                if temp_video.exists():
                    temp_video.unlink()
                    print(f"🧹 Cleaned up temp video: {temp_video.name}")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"⚠️ Could not delete temp video: {e}")

    async def extract_frame(self, video_path: str, time_sec: float = 0) -> np.ndarray:
        """
        Extract a single frame from video
        
        Args:
            video_path: Path to video file
            time_sec: Time in seconds to extract frame from
            
        Returns:
            Frame as numpy array (BGR format)
        """
        video_path = str(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Seek to position
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frame_no = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        
        ret, frame = cap.read()
        cap.release()
        
        # Fallback to first frame if seek failed
        if not ret or frame is None:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise ValueError("Could not extract frame from video")
                
        return frame

    def _log_error(self, input_path: Path, error: Exception) -> None:
        """Log error to file"""
        try:
            import datetime
            import traceback
            
            with open("error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now()}] Error processing {input_path}: {str(error)}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "=" * 50 + "\n")
                
            print(f"❌ Error logged: {error}")
        except Exception:
            pass

    def _cleanup_temp_dirs(self, *dirs: Path) -> None:
        """Clean up temporary directories"""
        for dir_path in dirs:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
            except Exception:
                pass
