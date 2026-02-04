// Queue item card component

import type { MediaFile } from '../types'
import { formatSize, getStatusText, getStatusColor } from '../utils/helpers'
import { API_URL } from '../config'

interface QueueItemProps {
  file: MediaFile
  isSelected: boolean
  onSelect: () => void
  onRemove: () => void
}

export function QueueItem({ file, isSelected, onSelect, onRemove }: QueueItemProps) {
  return (
    <div 
      onClick={onSelect}
      className={`group flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 animate-slide-in cursor-pointer
        ${isSelected 
          ? 'bg-indigo-500/10 border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.1)]' 
          : 'bg-white/2 hover:bg-white/4 border-white/5'
        }`}
    >
      {/* Thumbnail */}
      <div className="w-16 h-12 bg-black/40 rounded-lg flex items-center justify-center text-zinc-600 shrink-0 border border-white/5">
        {file.fileType === 'image' ? (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        ) : (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="4" width="20" height="16" rx="2" />
            <polygon points="10 9 15 12 10 15 10 9" />
          </svg>
        )}
      </div>
      
      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-1">
          <div className="font-medium text-zinc-200 truncate flex-1" title={file.name}>
            {file.name}
          </div>
        </div>
        
        <div className="flex items-center gap-3 text-xs text-zinc-500">
          <span className="uppercase font-semibold tracking-wider text-zinc-600">
            {file.fileType}
          </span>
          <div className="w-1 h-1 rounded-full bg-white/20" />
          <span>{formatSize(file.size)}</span>
          {file.upscaleMethod && (
            <>
              <div className="w-1 h-1 rounded-full bg-white/20" />
              <span className={`font-bold ${file.upscaleMethod === 'Real-ESRGAN' ? 'text-indigo-400' : 'text-zinc-500'}`}>
                {file.upscaleMethod}
              </span>
            </>
          )}
        </div>
        
        {/* Progress Bar */}
        {(file.status === 'uploading' || file.status === 'processing') && (
          <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden relative">
            <div 
              className="h-full bg-linear-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-300 relative overflow-hidden" 
              style={{ width: `${file.progress}%` }}
            >
              <div className="absolute inset-0 animate-shimmer" />
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4">
        <span className={`text-xs font-bold whitespace-nowrap ${getStatusColor(file.status)} ${file.status === 'processing' ? 'animate-pulse' : ''}`}>
          {file.status === 'processing' && file.message ? file.message : getStatusText(file.status)}
        </span>

        <div className="flex items-center gap-2">
          {/* Download button */}
          {file.status === 'completed' && file.jobId && (
            <a 
              href={`${API_URL}/api/download/${file.jobId}`}
              download
              onClick={(e) => e.stopPropagation()}
              className="p-2 text-emerald-400 hover:text-emerald-300 hover:bg-emerald-400/10 rounded-lg transition"
              title="Download"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </a>
          )}
          
          {/* Remove button */}
          <button 
            onClick={(e) => { e.stopPropagation(); onRemove(); }} 
            className="p-2 text-zinc-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition cursor-pointer" 
            title="Remove"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
