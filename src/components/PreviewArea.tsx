import type { MediaFile, PreviewImage } from '../types'
import { ImageComparison } from './ImageComparison'

interface PreviewAreaProps {
  files: MediaFile[]
  selectedFile: MediaFile | undefined
  previewImages: Record<string, PreviewImage>
  loadingPreview: string | null
}

export function PreviewArea({ files, selectedFile, previewImages, loadingPreview }: PreviewAreaProps) {
  return (
    <div className="flex-1 min-h-[350px] bg-[#18181b] rounded-3xl border border-white/10 flex items-center justify-center relative overflow-hidden group shadow-md">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,var(--tw-gradient-stops))] from-indigo-900/10 via-[#18181b] to-[#18181b]" />
      
      {files.length === 0 ? (
        <EmptyState />
      ) : selectedFile ? (
        <SelectedFilePreview 
          file={selectedFile} 
          previewImages={previewImages} 
          loadingPreview={loadingPreview} 
        />
      ) : (
        <QueuedFilesState count={files.length} />
      )}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="relative text-center p-12">
      <div className="w-24 h-24 mx-auto mb-6 rounded-3xl bg-white/2 border border-white/5 flex items-center justify-center group-hover:scale-105 transition-transform duration-500">
        <svg className="w-10 h-10 text-zinc-600 group-hover:text-indigo-500/50 transition-colors" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
      </div>
      <h3 className="text-xl font-medium text-zinc-200 mb-2">No Active Preview</h3>
      <p className="text-zinc-500 max-w-xs mx-auto leading-relaxed">
        Select a file from the queue or upload new media to see the preview
      </p>
    </div>
  )
}

function QueuedFilesState({ count }: { count: number }) {
  return (
    <div className="relative text-center p-12">
      <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-indigo-500/10 flex items-center justify-center ring-1 ring-indigo-500/30 animate-pulse">
        <svg className="w-10 h-10 text-indigo-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      </div>
      <h3 className="text-2xl font-bold text-white mb-2">{count} Files Queued</h3>
      <p className="text-indigo-300 font-medium">Select a file to preview</p>
    </div>
  )
}

interface SelectedFilePreviewProps {
  file: MediaFile
  previewImages: Record<string, PreviewImage>
  loadingPreview: string | null
}

function SelectedFilePreview({ file, previewImages, loadingPreview }: SelectedFilePreviewProps) {
  const preview = previewImages[file.id]

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-black/50">
      {preview ? (
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Hidden image to maintain size */}
          <img 
            src={URL.createObjectURL(file.file)}
            alt=""
            className="w-full h-full object-contain max-h-[350px] invisible"
          />
          {/* ImageComparison positioned absolutely */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-full h-full max-h-[350px] flex items-center justify-center">
              <ImageComparison 
                originalImage={preview.original}
                upscaledImage={preview.upscaled}
                label={preview.method ? `Preview (${preview.method})` : 'Preview Frame'}
              />
            </div>
          </div>
        </div>
      ) : (
        <>
          {file.fileType === 'video' ? (
            <video 
              src={URL.createObjectURL(file.file)} 
              className="w-full h-full object-contain max-h-[350px]"
              controls
              muted
            />
          ) : (
            <img 
              src={URL.createObjectURL(file.file)}
              alt={file.name}
              className="w-full h-full object-contain max-h-[350px]"
            />
          )}
          
          {/* Overlay Info */}
          <div className="absolute top-4 left-4 right-4 flex justify-between pointer-events-none z-10">
            <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur text-xs font-medium text-white border border-white/10">
              {file.name}
            </div>
          </div>

          {/* Loading Preview Indicator */}
          {loadingPreview === file.id && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20">
              <div className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-black/60 backdrop-blur text-white font-medium">
                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating Preview...
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
