// Video Upscaler - Main Application
// Refactored to use modular components

import { useState, useEffect } from 'react'
import type { HistoryItem } from './types'

import { useUpscaler } from './hooks/useUpscaler'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { PreviewArea } from './components/PreviewArea'
import { QueueItem } from './components/QueueItem'
import { LibraryView } from './components/LibraryView'
import { MediaViewer } from './components/MediaViewer'

function App() {
  // Tab state with localStorage persistence
  const [activeTab, setActiveTab] = useState<'workspace' | 'library'>(() => {
    return (localStorage.getItem('upscaler_active_tab') as 'workspace' | 'library') || 'workspace'
  })
  
  // Media viewer state
  const [viewingMedia, setViewingMedia] = useState<HistoryItem | null>(null)

  // Upscaler hook - contains all business logic and state
  const {
    files,
    selectedFile,
    selectedFileId,
    resolution,
    previewImages,
    loadingPreview,
    modelStatus,
    history,
    setSelectedFileId,
    setResolution,
    addFiles,
    removeFile,
    startAll,
    setHistory
  } = useUpscaler()

  // Persist active tab
  useEffect(() => {
    localStorage.setItem('upscaler_active_tab', activeTab)
  }, [activeTab])

  const hasReadyFiles = files.some(f => f.status === 'ready')

  return (
    <div className="min-h-screen bg-[#0f0f13] text-zinc-100 flex flex-col font-sans selection:bg-indigo-500/30">
      {/* Header */}
      <Header 
        activeTab={activeTab}
        onTabChange={setActiveTab}
        modelStatus={modelStatus}
      />

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {activeTab === 'workspace' ? (
          <>
            {/* Sidebar */}
            <Sidebar
              resolution={resolution}
              onResolutionChange={setResolution}
              onFilesAdd={addFiles}
              onStartAll={startAll}
              hasReadyFiles={hasReadyFiles}
            />

            {/* Main Content */}
            <main className="flex-1 p-6 overflow-y-auto flex flex-col gap-6 bg-[#0f0f13] custom-scrollbar">
              {/* Preview Area */}
              <PreviewArea
                files={files}
                selectedFile={selectedFile}
                previewImages={previewImages}
                loadingPreview={loadingPreview}
              />

              {/* Queue Section */}
              {files.length > 0 && (
                <div className="bg-[#18181b] rounded-2xl border border-white/5 p-6 shadow-2xl shadow-black/50">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <h3 className="font-bold text-lg text-zinc-200">Processing Queue</h3>
                      <span className="px-2.5 py-0.5 rounded-full bg-white/5 text-xs font-medium text-zinc-400 border border-white/5">
                        {files.length}
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid gap-3 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                    {files.map(file => (
                      <QueueItem
                        key={file.id}
                        file={file}
                        isSelected={selectedFileId === file.id}
                        onSelect={() => setSelectedFileId(file.id)}
                        onRemove={() => removeFile(file.id)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </main>
          </>
        ) : (
          <LibraryView 
            history={history} 
            onHistoryChange={setHistory} 
            onView={setViewingMedia} 
          />
        )}
      </div>

      {/* Full Screen Viewer */}
      <MediaViewer item={viewingMedia} onClose={() => setViewingMedia(null)} />
    </div>
  )
}

export default App