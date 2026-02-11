import type { HistoryItem } from '../types'

interface DeleteConfirmationModalProps {
  isOpen: boolean
  item: HistoryItem | null
  onClose: () => void
  onConfirm: () => void
}

export function DeleteConfirmationModal({ isOpen, item, onClose, onConfirm }: DeleteConfirmationModalProps) {
  if (!isOpen || !item) return null

  return (
    <div 
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in p-4" 
      onClick={onClose}
    >
      <div 
        className="bg-[#121214] border border-white/10 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 animate-scale-up" 
        onClick={e => e.stopPropagation()}
      >
        <div className="flex flex-col items-center text-center gap-4">
          <div className="w-12 h-12 rounded-full bg-red-500/10 text-red-500 flex items-center justify-center mb-1">
            <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18" />
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
              <path d="M8 6V4c0-1 1-1 1-1h6c1 0 1 1 1 1v2" />
              <line x1="10" y1="11" x2="10" y2="17" />
              <line x1="14" y1="11" x2="14" y2="17" />
            </svg>
          </div>
          
          <div>
            <h3 className="text-lg font-bold text-white mb-2">Delete File?</h3>
            <p className="text-zinc-400 text-sm">
              Are you sure you want to delete <span className="text-white font-medium">"{item.name}"</span>? 
              This action cannot be undone.
            </p>
          </div>

          <div className="flex w-full gap-3 mt-2">
            <button 
              onClick={onClose}
              className="flex-1 px-4 py-2.5 rounded-xl bg-white/5 text-zinc-300 font-medium hover:bg-white/10 transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button 
              onClick={onConfirm}
              className="flex-1 px-4 py-2.5 rounded-xl bg-red-500 text-white font-bold hover:bg-red-600 transition-colors shadow-lg shadow-red-500/20 cursor-pointer"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
