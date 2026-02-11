import { useState } from 'react'
import type { HistoryItem } from '../types'
import { HistoryCard } from './HistoryCard'
import { DeleteConfirmationModal } from './DeleteConfirmationModal'
import { API_URL } from '../config'

interface LibraryViewProps {
  history: HistoryItem[]
  onHistoryChange: React.Dispatch<React.SetStateAction<HistoryItem[]>>
  onView: (item: HistoryItem) => void
}

export function LibraryView({ history, onHistoryChange, onView }: LibraryViewProps) {
  const images = history.filter(item => item.fileType === 'image')
  const videos = history.filter(item => item.fileType === 'video')
  const [itemToDelete, setItemToDelete] = useState<HistoryItem | null>(null)

  const handleDeleteConfirm = async () => {
    if (!itemToDelete) return
    
    try {
      await fetch(`${API_URL}/api/jobs/${itemToDelete.jobId}`, { method: 'DELETE' })
      onHistoryChange(prev => prev.filter(h => h.id !== itemToDelete.id))
      setItemToDelete(null)
    } catch (err) {
      console.error('Failed to delete job', err)
    }
  }

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-[#18181b] animate-slide-in custom-scrollbar">
      <div className="max-w-6xl mx-auto space-y-12 pb-20 relative">
        {history.length === 0 ? (
          <EmptyLibraryState />
        ) : (
          <>
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
              <div>
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-linear-to-r from-white to-white/60">
                  Media Vault
                </h2>
                <p className="text-zinc-400 mt-1">Your collection of AI-enhanced assets</p>
              </div>
            </div>

            {/* Images Category */}
            {images.length > 0 && (
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center text-indigo-400 border border-indigo-500/20">
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-zinc-200">
                    Images <span className="text-zinc-500 font-medium ml-2 text-sm">({images.length})</span>
                  </h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {images.map(item => (
                    <HistoryCard 
                      key={item.id} 
                      item={item} 
                      onDelete={() => setItemToDelete(item)} 
                      onView={onView} 
                    />
                  ))}
                </div>
              </section>
            )}

            {/* Videos Category */}
            {videos.length > 0 && (
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 border border-purple-500/20">
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="2" y="4" width="20" height="16" rx="2" />
                      <polygon points="10 9 15 12 10 15 10 9" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-zinc-200">
                    Videos <span className="text-zinc-500 font-medium ml-2 text-sm">({videos.length})</span>
                  </h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {videos.map(item => (
                    <HistoryCard 
                      key={item.id} 
                      item={item} 
                      onDelete={() => setItemToDelete(item)} 
                      onView={onView} 
                    />
                  ))}
                </div>
              </section>
            )}
          </>
        )}
        
        {/* Delete Confirmation Modal */}
        <DeleteConfirmationModal 
          isOpen={!!itemToDelete} 
          item={itemToDelete} 
          onClose={() => setItemToDelete(null)} 
          onConfirm={handleDeleteConfirm} 
        />
      </div>
    </div>
  )
}

function EmptyLibraryState() {
  return (
    <div className="w-full flex flex-col items-center justify-center py-24 border border-dashed border-white/5 rounded-3xl bg-white/2">
      <div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center text-zinc-700 mb-6">
        <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
      </div>
      <h3 className="text-xl font-bold text-zinc-400 mb-2">Your vault is empty</h3>
      <p className="text-zinc-600 max-w-xs text-center">
        Successfully upscaled items will automatically appear here for future access.
      </p>
    </div>
  )
}
