import { useState } from 'react'
import type { HistoryItem } from '../types'
import { API_URL } from '../config'

interface HistoryCardProps {
  item: HistoryItem
  onDelete: () => void
  onView: (item: HistoryItem) => void
}

export function HistoryCard({ item, onDelete, onView }: HistoryCardProps) {
  const [error, setError] = useState(false)

  return (
    <div className="glass-panel group p-4 rounded-2xl hover:border-indigo-500/50 transition-all duration-300 bg-white/5">
      <div className="flex gap-4 mb-4">
        {/* Media Preview */}
        <div 
          onClick={() => onView(item)}
          className="w-24 h-24 rounded-xl bg-black/40 flex items-center justify-center text-zinc-600 shrink-0 border border-white/5 overflow-hidden relative cursor-pointer hover:ring-2 hover:ring-indigo-500/50 transition-all group/media"
        >
          {!error ? (
            item.fileType === 'image' ? (
              <img 
                src={`${API_URL}/api/download/${item.jobId}`} 
                alt={item.name} 
                className="w-full h-full object-cover transition-transform duration-500 group-hover/media:scale-110"
                onError={() => setError(true)}
              />
            ) : (
              <video 
                src={`${API_URL}/api/download/${item.jobId}`}
                className="w-full h-full object-cover transition-transform duration-500 group-hover/media:scale-110"
                muted
                loop
                playsInline
                onMouseOver={e => e.currentTarget.play().catch(() => {})}
                onMouseOut={e => {
                  e.currentTarget.pause()
                  e.currentTarget.currentTime = 0
                }}
                onError={() => setError(true)}
              />
            )
          ) : (
            // Fallback Icon
            item.fileType === 'image' ? (
              <svg className="w-8 h-8 opacity-50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            ) : (
              <svg className="w-8 h-8 opacity-50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <polygon points="10 9 15 12 10 15 10 9" />
              </svg>
            )
          )}
        </div>

        <div className="flex-1 min-w-0 flex flex-col justify-between py-1">
          <div>
            <div className="font-bold text-zinc-200 truncate mb-1 leading-tight" title={item.name}>
              {item.name}
            </div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] font-bold text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded border border-indigo-500/10 uppercase tracking-tight">
                {item.resolution}p
              </span>
              <span className="text-[10px] text-zinc-500 font-medium px-1 border-l border-white/10 uppercase">
                {new Date(item.timestamp).toLocaleDateString()}
              </span>
            </div>
            {/* Upscale Method Badge */}
            {item.upscaleMethod && (
              <div className="inline-block">
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border uppercase tracking-wide
                  ${item.upscaleMethod === 'Real-ESRGAN' 
                    ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10' 
                    : 'text-zinc-500 border-zinc-700 bg-zinc-800'
                  }`}
                >
                  {item.upscaleMethod}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="flex items-center justify-between pt-3 border-t border-white/5">
        <a 
          href={`${API_URL}/api/download/${item.jobId}`}
          download
          className="flex items-center gap-2 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download
        </a>
        
        <button 
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 hover:text-red-400 transition-colors cursor-pointer"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          </svg>
          Delete
        </button>
      </div>
    </div>
  )
}
