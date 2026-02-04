# ⚡ AI Video Upscaler

![Project Banner](public/app-logo.png)

A modern, web-based application for upscaling videos and images using AI (Real-ESRGAN). Built with React, TypeScript, Tailwind CSS, and FastAPI.

## ✨ Features

- **🎬 AI Video Upscaling**: Upscale videos to 2x or 4x resolution.
- **🖼️ Image Enhancement**: Instantly upscale images with AI details restoration.
- **⚡ Real-time Progress**: Track upscaling job progress via WebSockets.
- **💾 Media Vault**: History of your upscaled files with preview and download.
- **🎨 Modern UI**: Beautiful Glassmorphism design with dark mode and smooth animations.
- **🔒 Secure**: Built-in file size limits, path validation, and filename sanitization.

## 🛠️ Tech Stack

**Frontend:**

- React + Vite
- TypeScript
- Tailwind CSS (v4)
- Framer Motion (animations via CSS)

**Backend:**

- Python + FastAPI
- Real-ESRGAN (AI Engine)
- OpenCV
- FFmpeg (Video processing)

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.10+)
- **FFmpeg** installed and added to system PATH
- CUDA (Optional, but recommended for faster upscaling with NVIDIA GPUs)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/video-upscaler.git
   cd video-upscaler
   ```

2. **Backend Setup**

   ```bash
   cd backend
   python -m venv venv

   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   Open a new terminal in the root directory:
   ```bash
   npm install
   ```

### Running the App

1. **Start Backend**

   ```bash
   cd backend
   # Make sure venv is activated
   python -m uvicorn main:app --reload
   ```

2. **Start Frontend**

   ```bash
   # In root directory
   npm run dev
   ```

3. Open your browser at `http://localhost:5173`

## 📦 Project Structure

```
├── src/                # Frontend Source
│   ├── components/     # UI Components
│   ├── hooks/          # Custom React Hooks
│   ├── types/          # TypeScript Definitions
│   └── utils/          # Helper Functions
├── backend/            # Backend Source
│   ├── main.py         # API Server & Endpoints
│   ├── upscaler.py     # AI Logic & FFmpeg handler
│   └── queue_manager.py# Job Queue System
└── public/             # Static Assets
```

## 📝 License

This project is open source.

---