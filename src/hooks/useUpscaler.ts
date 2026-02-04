// Custom hook for upscaler state management

import { useState, useCallback, useEffect, useMemo } from 'react'
import type { MediaFile, HistoryItem, Resolution, ModelStatus, PreviewImage } from '../types'
import { API_URL } from '../config'
import { generateId } from '../utils/helpers'

interface ServerJob {
  id: string
  original_filename: string
  status: string
  completed_at: string
  target_resolution?: number
  scale: number
  upscale_method?: string
}

export function useUpscaler() {
  // File management state
  const [files, setFiles] = useState<MediaFile[]>([])
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null)
  
  // Settings state
  const [resolution, setResolution] = useState<Resolution>(1440)
  
  // Preview state
  const [previewImages, setPreviewImages] = useState<Record<string, PreviewImage>>({})
  const [loadingPreview, setLoadingPreview] = useState<string | null>(null)
  
  // Model status
  const [modelStatus, setModelStatus] = useState<ModelStatus>('offline')
  
  // History (lazy initialization)
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem('upscaler_history')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  // Selected file computed value
  const selectedFile = useMemo(
    () => files.find(f => f.id === selectedFileId),
    [files, selectedFileId]
  )

  // Poll model status
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
    const interval = setInterval(checkStatus, 3000)
    return () => clearInterval(interval)
  }, [])

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('upscaler_history', JSON.stringify(history))
  }, [history])

  // Sync history with backend
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs`)
        if (res.ok) {
          const data = await res.json()
          const serverJobs = data.jobs.filter((j: ServerJob) => j.status === 'completed')
          
          const newHistory: HistoryItem[] = serverJobs.map((j: ServerJob) => ({
            id: j.id,
            jobId: j.id,
            name: j.original_filename,
            fileType: /\.(mp4|avi|mov|mkv|webm)$/i.test(j.original_filename) ? 'video' : 'image',
            timestamp: new Date(j.completed_at).getTime(),
            resolution: j.target_resolution || (j.scale === 4 ? 2160 : 1080),
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

  // Generate preview
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

  // Auto-generate preview when file is selected and ready
  useEffect(() => {
    if (selectedFile && selectedFile.status === 'ready' && selectedFile.fileId && !previewImages[selectedFile.id]) {
      generatePreview(selectedFile)
    }
  }, [selectedFile, generatePreview, previewImages])

  // Upload file to server
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

  // Add files to queue
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

  // Start upscaling a file
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
      
      // Store job_id
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, jobId: data.job_id } : v
      ))
      
      // WebSocket for progress
      const ws = new WebSocket(`ws://localhost:8000/ws/progress/${data.job_id}`)
      
      ws.onmessage = (event) => {
        const progress = JSON.parse(event.data)
        
        if (progress.status === 'completed') {
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
        }

        setFiles(prev => prev.map(v => {
          if (v.id === mediaFile.id) {
            if (progress.status === 'completed') {
              return { ...v, status: 'completed' as const, progress: 100, outputPath: progress.output_path, upscaleMethod: progress.upscale_method }
            }
            if (progress.status === 'failed') {
              return { ...v, status: 'error' as const, error: progress.message }
            }
            return { ...v, progress: progress.progress, message: progress.message }
          }
          return v
        }))
        
        if (progress.status === 'completed' || progress.status === 'failed') {
          ws.close()
        }
      }
      
      ws.onerror = () => {
        setFiles(prev => prev.map(v =>
          v.id === mediaFile.id ? { ...v, status: 'error' as const, error: 'การเชื่อมต่อล้มเหลว' } : v
        ))
      }
    } catch {
      setFiles(prev => prev.map(v =>
        v.id === mediaFile.id ? { ...v, status: 'error' as const, error: 'เกิดข้อผิดพลาด' } : v
      ))
    }
  }, [modelStatus, resolution])

  // Start all ready files
  const startAll = useCallback(() => {
    if (modelStatus !== 'ready') return
    files.filter(v => v.status === 'ready').forEach(file => startUpscale(file))
  }, [files, modelStatus, startUpscale])

  // Remove file from queue
  const removeFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(v => v.id !== id))
    if (selectedFileId === id) {
      setSelectedFileId(null)
    }
  }, [selectedFileId])

  // Delete history item
  const deleteHistoryItem = useCallback(async (item: HistoryItem) => {
    try {
      await fetch(`${API_URL}/api/jobs/${item.jobId}`, { method: 'DELETE' })
      setHistory(prev => prev.filter(h => h.id !== item.id))
    } catch (err) {
      console.error('Failed to delete job', err)
    }
  }, [])

  // Clear all history
  const clearHistory = useCallback(() => {
    setHistory([])
  }, [])

  return {
    // State
    files,
    selectedFile,
    selectedFileId,
    resolution,
    previewImages,
    loadingPreview,
    modelStatus,
    history,
    
    // Actions
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
