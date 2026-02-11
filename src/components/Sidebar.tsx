import type { Resolution } from '../types'
import { UploadZone } from './UploadZone'
import { ResolutionSelector } from './ResolutionSelector'

interface SidebarProps {
  resolution: Resolution
  onResolutionChange: (resolution: Resolution) => void
  onFilesAdd: (files: File[]) => void
  onStartAll: () => void
  hasReadyFiles: boolean
}

export function Sidebar({ 
  resolution, 
  onResolutionChange, 
  onFilesAdd, 
  onStartAll, 
  hasReadyFiles 
}: SidebarProps) {
  return (
    <aside className="w-80 min-w-80 bg-[#18181b] border-r border-white/10 p-6 overflow-y-auto flex flex-col gap-8 shadow-sm">
      {/* Upload Section */}
      <UploadZone onFilesAdd={onFilesAdd} />

      <div className="h-px bg-linear-to-r from-transparent via-white/10 to-transparent" />

      {/* Resolution Section */}
      <ResolutionSelector 
        resolution={resolution} 
        onResolutionChange={onResolutionChange} 
      />

      {/* Start Button */}
      <div className="mt-auto">
        <button 
          onClick={onStartAll}
          disabled={!hasReadyFiles}
          className="group w-full py-4 rounded-xl font-bold text-sm transition-all duration-300 flex items-center justify-center gap-2.5 relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          <div className="absolute inset-0 bg-linear-to-r from-indigo-600 to-purple-600 opacity-90 group-hover:opacity-100 transition-opacity" />
          <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
          <svg className="relative w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          <span className="relative text-white tracking-wide">Start Process</span>
        </button>
      </div>
    </aside>
  )
}
