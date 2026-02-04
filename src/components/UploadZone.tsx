// Upload zone with drag & drop functionality

import { useState, useCallback } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface UploadZoneProps {
  onFilesAdd: (files: File[]) => void
}

export function UploadZone({ onFilesAdd }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDrop = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = Array.from(e.dataTransfer.files).filter(file => 
      file.type.startsWith('video/') || file.type.startsWith('image/')
    )
    onFilesAdd(droppedFiles)
  }, [onFilesAdd])

  const handleDragOver = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).filter(file => 
        file.type.startsWith('video/') || file.type.startsWith('image/')
      )
      onFilesAdd(selectedFiles)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        Input Files
      </div>
      <div 
        className={`group relative p-8 border border-dashed rounded-2xl text-center cursor-pointer transition-all duration-300
          ${isDragging 
            ? 'border-indigo-500 bg-indigo-500/10' 
            : 'border-white/10 bg-white/2 hover:border-indigo-500/50 hover:bg-white/4'
          }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className={`w-12 h-12 mx-auto mb-4 rounded-xl flex items-center justify-center transition-all duration-300
          ${isDragging 
            ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' 
            : 'bg-white/5 text-zinc-400 group-hover:scale-110 group-hover:bg-indigo-500 group-hover:text-white'
          }`}
        >
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <h3 className="font-medium text-zinc-200 mb-1">Upload Media</h3>
        <p className="text-xs text-zinc-500 mb-3">Drag & Drop or Click</p>
        <div className="flex flex-wrap justify-center gap-1.5">
          {['JPG', 'PNG', 'MP4', 'MKV'].map(ext => (
            <span key={ext} className="text-[10px] font-medium text-zinc-500 bg-white/5 px-1.5 py-0.5 rounded border border-white/5">
              {ext}
            </span>
          ))}
        </div>
        <input 
          type="file" 
          accept="image/*,video/*" 
          multiple 
          onChange={handleFileInput}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
      </div>
    </div>
  )
}
