import { useCallback, useEffect, useMemo, useState } from 'react'
import { API_URL } from '../config'
import type { HistoryItem, MediaFile, ModelStatus, PreviewImage, Resolution } from '../types'
import { generateId } from '../utils/helpers'

const POLL_STATUS_MS = 3000
const POLL_JOB_MS = 1000

interface ServerJob {
  id: string
  original_filename: string
  status: string
  completed_at: string
  target_resolution?: number
  scale: number
  upscale_method?: string
}

interface ProgressUpdate {
  status: string
  progress?: number
  message?: string
  output_path?: string
  upscale_method?: string
}

function loadHistory(): HistoryItem[] {
  try {
    const saved = localStorage.getItem('upscaler_history')
    return saved ? JSON.parse(saved) : []
  } catch {
    return []
  }
}

export function useUpscaler() {
  const [files, setFiles] = useState<MediaFile[]>([])
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null)
  const [resolution, setResolution] = useState<Resolution>(1440)
  const [previewImages, setPreviewImages] = useState<Record<string, PreviewImage>>({})
  const [loadingPreview, setLoadingPreview] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus>('offline')
  const [history, setHistory] = useState<HistoryItem[]>(loadHistory)

  const selectedFile = useMemo(
    () => files.find(f => f.id === selectedFileId),
    [files, selectedFileId]
  )

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/status`)
        if (res.ok) {
          const data = await res.json()
          setModelStatus(data.status)
        } else {
          setModelStatus('offline')
        }
      } catch {
        setModelStatus('offline')
      }
    }
    
    checkStatus()
    const interval = setInterval(checkStatus, POLL_STATUS_MS)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    localStorage.setItem('upscaler_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs`)
        if (res.ok) {
          const data = await res.json()
          const serverJobs = data.jobs.filter((j: ServerJob) => j.status === 'completed')
          
          const isVideo = /\.(mp4|avi|mov|mkv|webm)$/i
          const newHistory: HistoryItem[] = serverJobs.map((j: ServerJob) => ({
            id: j.id,
            jobId: j.id,
            name: j.original_filename,
            fileType: isVideo.test(j.original_filename) ? 'video' : 'image',
            timestamp: new Date(j.completed_at).getTime(),
            resolution: j.target_resolution ?? (j.scale === 4 ? 2160 : 1080),
            upscaleMethod: j.upscale_method
          }))

          newHistory.sort((a, b) => b.timestamp - a.timestamp)
          setHistory(newHistory)
        }
      } catch (e) {
        console.error('Failed to sync history', e)
      }
    }
    fetchJobs()
  }, [])

  const generatePreview = useCallback(async (file: MediaFile) => {
    if (!file.fileId || previewImages[file.id] || loadingPreview === file.id) return
    
    setLoadingPreview(file.id)
    try {
      const response = await fetch(`${API_URL}/api/preview/${file.fileId}`)
      if (!response.ok) throw new Error('Preview failed')
      const data = await response.json()
      setPreviewImages(prev => ({
        ...prev,
        [file.id]: {
          original: `${API_URL}${data.original}`,
          upscaled: `${API_URL}${data.upscaled}`,
          method: data.method
        }
      }))
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingPreview(null)
    }
  }, [previewImages, loadingPreview])

  useEffect(() => {
    if (selectedFile && selectedFile.status === 'ready' && selectedFile.fileId && !previewImages[selectedFile.id]) {
      generatePreview(selectedFile)
    }
  }, [selectedFile, generatePreview, previewImages])

  const uploadFile = useCallback(async (mediaFile: MediaFile) => {
    setFiles(prev => prev.map(v =>
      v.id === mediaFile.id ? { ...v, status: 'uploading' as const, progress: 0 } : v
    ))
    try {
      const formData = new FormData()
      formData.append('file', mediaFile.file)
      const response = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData })
      if (!response.ok) throw new Error('Upload failed')
      const data = await response.json()
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, status: 'ready' as const, progress: 100, serverPath: data.path, fileId: data.file_id } : v
      ))
    } catch {
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, status: 'error' as const, error: 'อัพโหลดไม่สำเร็จ' } : v
      ))
    }
  }, [])

  const addFiles = useCallback((fileList: File[]) => {
    const newFiles: MediaFile[] = fileList.map(file => ({
      id: generateId(),
      file,
      name: file.name,
      size: file.size,
      fileType: file.type.startsWith('image/') ? 'image' : 'video',
      status: 'pending',
      progress: 0
    }))
    setFiles(prev => [...prev, ...newFiles])
    if (newFiles.length > 0) {
      setSelectedFileId(newFiles[0].id)
    }
    newFiles.forEach(mediaFile => uploadFile(mediaFile))
  }, [uploadFile])

  const startUpscale = useCallback(async (mediaFile: MediaFile) => {
    if (!mediaFile.serverPath || modelStatus !== 'ready') return
    
    setFiles(prev => prev.map(v =>
      v.id === mediaFile.id ? { ...v, status: 'processing' as const, progress: 0 } : v
    ))
    
    try {
      const response = await fetch(
        `${API_URL}/api/upscale?file_path=${encodeURIComponent(mediaFile.serverPath)}&target_resolution=${resolution}&original_filename=${encodeURIComponent(mediaFile.name)}`,
        { method: 'POST' }
      )
      if (!response.ok) throw new Error('Upscale failed')
      const data = await response.json()
      
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, jobId: data.job_id } : v
      ))

      let isFinished = false
      let pollingInterval: ReturnType<typeof setInterval> | null = null

      const handleProgress = (progress: ProgressUpdate) => {
        if (isFinished) return

        if (progress.status === 'completed') {
          isFinished = true
          setHistory(prev => {
            if (prev.some(item => item.jobId === data.job_id)) return prev
            return [{
              id: Math.random().toString(36).substring(2, 9),
              name: mediaFile.name,
              jobId: data.job_id,
              fileType: mediaFile.fileType,
              timestamp: Date.now(),
              resolution: resolution,
              upscaleMethod: progress.upscale_method
            }, ...prev]
          })

          if (pollingInterval !== null) {
            clearInterval(pollingInterval)
            pollingInterval = null
          }
        }

        setFiles(prev => prev.map(v => {
          if (v.id === mediaFile.id) {
            if (progress.status === 'completed') {
              return {
                ...v,
                status: 'completed' as const,
                progress: 100,
                outputPath: progress.output_path,
                upscaleMethod: progress.upscale_method
              }
            }
            if (progress.status === 'failed') {
              return { ...v, status: 'error' as const, error: progress.message }
            }
            return {
              ...v,
              progress: progress.progress ?? v.progress ?? 0,
              message: progress.message
            }
          }
          return v
        }))
      }

      const startPolling = () => {
        if (isFinished || pollingInterval !== null) return

        pollingInterval = setInterval(async () => {
          try {
            const res = await fetch(`${API_URL}/api/jobs/${data.job_id}`)
            if (!res.ok) return
            const job = await res.json()

            const progress = {
              status: job.status,
              progress: job.progress ?? (job.status === 'completed' ? 100 : 0),
              message: job.error || '',
              output_path: job.output_path,
              upscale_method: job.upscale_method
            }

            handleProgress(progress)

            if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
              if (pollingInterval !== null) {
                clearInterval(pollingInterval)
                pollingInterval = null
              }
            }
          } catch {
            // retry next interval
          }
        }, POLL_JOB_MS)
      }
      
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsHost = window.location.host || 'localhost:8000'
      let ws: WebSocket | null = null

      try {
        ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/progress/${data.job_id}`)

        ws.onmessage = (event) => {
          const progress = JSON.parse(event.data)
          handleProgress(progress)

          if (progress.status === 'completed' || progress.status === 'failed') {
            ws?.close()
          }
        }

        ws.onerror = () => {
          ws?.close()
          startPolling()
        }

        ws.onclose = (evt) => {
          if (evt.code === 4030 || evt.code === 1006) {
            startPolling()
          }
        }
      } catch {
        startPolling()
      }
    } catch {
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, status: 'error' as const, error: 'เกิดข้อผิดพลาด' } : v
      ))
    }
  }, [modelStatus, resolution])

  const startAll = useCallback(() => {
    if (modelStatus !== 'ready') return
    files.filter(v => v.status === 'ready').forEach(file => startUpscale(file))
  }, [files, modelStatus, startUpscale])

  const removeFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(v => v.id !== id))
    if (selectedFileId === id) {
      setSelectedFileId(null)
    }
  }, [selectedFileId])

  const deleteHistoryItem = useCallback(async (item: HistoryItem) => {
    try {
      await fetch(`${API_URL}/api/jobs/${item.jobId}`, { method: 'DELETE' })
      setHistory(prev => prev.filter(h => h.id !== item.id))
    } catch (err) {
      console.error('Failed to delete job', err)
    }
  }, [])

  const clearHistory = useCallback(() => {
    setHistory([])
  }, [])

  return {
    files,
    selectedFile,
    selectedFileId,
    resolution,
    previewImages,
    loadingPreview,
    modelStatus,
    history,
    setSelectedFileId,
    setResolution,
    addFiles,
    removeFile,
    startUpscale,
    startAll,
    generatePreview,
    setHistory,
    deleteHistoryItem,
    clearHistory
  }
}
