// Header component with logo, navigation tabs, and model status

import type { ModelStatus } from '../types'

interface HeaderProps {
  activeTab: 'workspace' | 'library'
  onTabChange: (tab: 'workspace' | 'library') => void
  modelStatus: ModelStatus
}

export function Header({ activeTab, onTabChange, modelStatus }: HeaderProps) {
  return (
    <header className="h-16 px-6 border-b border-white/10 bg-[#18181b] flex items-center justify-between sticky top-0 z-50 shadow-sm">
      {/* Logo */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute -inset-1 bg-linear-to-r from-indigo-500 to-purple-500 rounded-full blur opacity-25"></div>
            <svg className="relative w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="url(#grad)" strokeWidth="2">
              <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#818cf8" />
                  <stop offset="100%" stopColor="#c084fc" />
                </linearGradient>
              </defs>
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-linear-to-r from-indigo-400 to-purple-400">
            Upscaler
          </h1>
        </div>
      </div>
      
      {/* Navigation Tabs */}
      <div className="flex items-center gap-1 bg-white/4 p-1 rounded-xl border border-white/5">
        <button 
          onClick={() => onTabChange('workspace')}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all cursor-pointer ${
            activeTab === 'workspace' 
              ? 'bg-indigo-600 text-white shadow-lg' 
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" />
          </svg>
          Workspace
        </button>
        <button 
          onClick={() => onTabChange('library')}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all cursor-pointer ${
            activeTab === 'library' 
              ? 'bg-indigo-600 text-white shadow-lg' 
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
          Vault
        </button>
      </div>

      {/* Model Status */}
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors duration-300
        ${modelStatus === 'ready' 
          ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
          : modelStatus === 'loading' 
            ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' 
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        }`}
      >
        <div className="relative flex h-2.5 w-2.5">
          <span className={`relative inline-flex rounded-full h-2.5 w-2.5 
            ${modelStatus === 'ready' 
              ? 'bg-emerald-500' 
              : modelStatus === 'loading' 
                ? 'bg-amber-500' 
                : 'bg-red-500'
            }`}
          />
        </div>
        <svg className="w-4 h-4 ml-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <line x1="8" y1="21" x2="16" y2="21" />
          <line x1="12" y1="17" x2="12" y2="21" />
        </svg>
        <span className="text-xs font-mono font-bold">
          {modelStatus === 'loading' ? 'LOADING...' : modelStatus === 'ready' ? 'READY' : 'OFFLINE'}
        </span>
      </div>
    </header>
  )
}
