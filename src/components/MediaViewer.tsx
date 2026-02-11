import type { HistoryItem } from '../types'
import { API_URL } from '../config'

interface MediaViewerProps {
  item: HistoryItem | null
  onClose: () => void
}

export function MediaViewer({ item, onClose }: MediaViewerProps) {
  if (!item) return null

  return (
    <div 
      className="fixed inset-0 z-100 flex items-center justify-center bg-black/95 backdrop-blur-xl animate-fade-in p-4 md:p-8" 
      onClick={onClose}
    >
      {/* Close Button */}
      <button 
        onClick={onClose}
        className="absolute top-4 right-4 md:top-6 md:right-6 p-2 rounded-full bg-white/10 text-white hover:bg-white/20 transition-all cursor-pointer z-50 hover:rotate-90 duration-300"
      >
        <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>

      <div className="relative w-full max-w-7xl h-full flex flex-col items-center justify-center gap-6" onClick={e => e.stopPropagation()}>
        
        {/* Media Container */}
        <div className="flex-1 w-full min-h-0 rounded-2xl overflow-hidden shadow-2xl border border-white/5 bg-black/50 flex items-center justify-center relative">
          {item.fileType === 'image' ? (
            <img 
              src={`${API_URL}/api/download/${item.jobId}`} 
              alt={item.name}
              className="max-w-full max-h-full object-contain"
            />
          ) : (
            <video 
              src={`${API_URL}/api/download/${item.jobId}`}
              className="max-w-full max-h-full object-contain"
              controls
              autoPlay
            />
          )}
        </div>
        
        {/* Footer Info */}
        <div className="shrink-0 text-center w-full max-w-4xl px-4 pb-2">
          <div className="flex items-center justify-center gap-4 mb-3">
            <h3 className="text-lg md:text-2xl font-bold text-white truncate leading-normal" title={item.name}>
              {item.name}
            </h3>
            <a 
              href={`${API_URL}/api/download/${item.jobId}`}
              download
              onClick={e => e.stopPropagation()}
              className="shrink-0 text-zinc-400 hover:text-indigo-400 transition-colors cursor-pointer p-2"
              title="Download Original"
            >
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </a>
          </div>
          
          <div className="inline-flex flex-wrap items-center justify-center gap-3 bg-white/5 px-4 py-2 rounded-full border border-white/5">
            <div className="flex items-center gap-2">
              {item.fileType === 'image' ? (
                <svg className="w-4 h-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              ) : (
                <svg className="w-4 h-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="2" y="4" width="20" height="16" rx="2" />
                  <polygon points="10 9 15 12 10 15 10 9" />
                </svg>
              )}
              <span className="text-sm font-medium text-zinc-300 uppercase">{item.fileType}</span>
            </div>
            <div className="w-1 h-1 rounded-full bg-zinc-600"></div>
            <span className="text-sm font-bold text-indigo-400 glow-text">{item.resolution}p</span>
            
            {item.upscaleMethod && (
              <>
                <div className="w-1 h-1 rounded-full bg-zinc-600"></div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded border uppercase tracking-wider
                  ${item.upscaleMethod === 'Real-ESRGAN' 
                    ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10' 
                    : 'text-zinc-400 border-zinc-700 bg-zinc-800'
                  }`}
                >
                  {item.upscaleMethod}
                </span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
