// Resolution selector grid

import type { Resolution } from '../types'
import { getResolutionLabel } from '../utils/helpers'

interface ResolutionSelectorProps {
  resolution: Resolution
  onResolutionChange: (resolution: Resolution) => void
}

const RESOLUTIONS: Resolution[] = [720, 1080, 1440, 2160]

export function ResolutionSelector({ resolution, onResolutionChange }: ResolutionSelectorProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <line x1="8" y1="21" x2="16" y2="21" />
          <line x1="12" y1="17" x2="12" y2="21" />
        </svg>
        Target Quality
      </div>
      <div className="grid grid-cols-2 gap-3">
        {RESOLUTIONS.map((res) => (
          <button
            key={res}
            onClick={() => onResolutionChange(res)}
            className={`group relative p-3 rounded-xl border text-sm font-medium transition-all duration-200 flex flex-col items-center gap-1 overflow-hidden cursor-pointer
              ${resolution === res 
                ? 'border-indigo-500/50 bg-indigo-500/10 text-indigo-300 shadow-[0_0_15px_rgba(99,102,241,0.1)]' 
                : 'border-white/5 bg-white/2 text-zinc-400 hover:border-white/10 hover:bg-white/4'
              }`}
          >
            {resolution === res && (
              <div className="absolute inset-0 bg-linear-to-tr from-indigo-500/10 to-transparent opacity-50" />
            )}
            <span className="relative text-lg font-bold">{res}p</span>
            <span className={`relative text-[10px] uppercase tracking-wider ${
              resolution === res ? 'text-indigo-400' : 'text-zinc-600'
            }`}>
              {getResolutionLabel(res)}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
