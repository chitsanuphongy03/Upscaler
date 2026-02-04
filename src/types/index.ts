// Types for Video Upscaler Application

export interface MediaFile {
  id: string
  file: File
  name: string
  size: number
  thumbnail?: string
  status: 'pending' | 'uploading' | 'ready' | 'processing' | 'completed' | 'error'
  progress: number
  serverPath?: string
  fileId?: string
  fileType: 'image' | 'video'
  outputPath?: string
  outputUrl?: string
  error?: string
  message?: string
  jobId?: string
  upscaleMethod?: string
}

export interface HistoryItem {
  id: string
  name: string
  jobId: string
  fileType: 'image' | 'video'
  timestamp: number
  resolution: number
  upscaleMethod?: string
}

export type Resolution = 720 | 1080 | 1440 | 2160

export type ModelStatus = 'offline' | 'loading' | 'ready' | 'failed'

export interface PreviewImage {
  original: string
  upscaled: string
  method?: string
}
