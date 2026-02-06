import type { MediaFile } from '../types'

export function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
}

/**
 * Get display text for file status
 */
export function getStatusText(status: MediaFile['status']): string {
  const statusMap: Record<MediaFile['status'], string> = {
    pending: 'Pending',
    uploading: 'Uploading...',
    ready: 'Ready',
    processing: 'Processing...',
    completed: 'Completed',
    error: '❌ Error'
  }
  return statusMap[status]
}

export function getStatusColor(status: MediaFile['status']): string {
  const colorMap: Record<MediaFile['status'], string> = {
    pending: 'text-zinc-500',
    uploading: 'text-white',
    ready: 'text-green-400',
    processing: 'text-indigo-400',
    completed: 'text-emerald-400',
    error: 'text-red-500'
  }
  return colorMap[status]
}

export function getResolutionLabel(resolution: number): string {
  const labels: Record<number, string> = {
    720: 'HD',
    1080: 'Full HD',
    1440: '2K QHD',
    2160: '4K UHD'
  }
  return labels[resolution] || ''
}

/**
 * Check if file is video based on extension
 */
export function isVideoFile(filename: string): boolean {
  return /\.(mp4|avi|mov|mkv|webm)$/i.test(filename)
}

export function generateId(): string {
  return crypto.randomUUID()
}
