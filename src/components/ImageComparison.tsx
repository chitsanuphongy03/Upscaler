import { useState, useRef, useEffect } from 'react'
import type { MouseEvent, TouchEvent } from 'react'
import '../App.css'

interface ImageComparisonProps {
  originalImage: string
  upscaledImage: string
  label?: string
}

export function ImageComparison({ originalImage, upscaledImage, label }: ImageComparisonProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [containerWidth, setContainerWidth] = useState(0) // Start with 0 or a safe default
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  const handleStart = () => {
    isDragging.current = true
  }

  const handleEnd = () => {
    isDragging.current = false
  }

  const handleMove = (clientX: number) => {
    if (!isDragging.current || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    const percent = (x / rect.width) * 100
    setSliderPosition(percent)
  }

  const onMouseMove = (e: MouseEvent) => handleMove(e.clientX)
  const onTouchMove = (e: TouchEvent) => handleMove(e.touches[0].clientX)

  // Global mouse up handler to catch drag end outside component
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      isDragging.current = false
    }
    window.addEventListener('mouseup', handleGlobalMouseUp)
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp)
  }, [])

  // ResizeObserver to track container width safely
  useEffect(() => {
    if (!containerRef.current) return
    
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
      }
    })
    
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full max-h-[400px] select-none overflow-hidden rounded-xl bg-black/50"
      onMouseDown={handleStart}
      onMouseMove={onMouseMove}
      onTouchStart={handleStart}
      onTouchMove={onTouchMove}
      onTouchEnd={handleEnd}
    >
      {/* Upscaled Image (Base) */}
      <img 
        src={upscaledImage} 
        alt="Upscaled" 
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
      />

      {/* Original Image (Overlay with Clip Path) */}
      <div 
        className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none border-r-2 border-white/50"
        style={{ width: `${sliderPosition}%` }}
      >
        <img 
          src={originalImage} 
          alt="Original" 
          className="absolute inset-0 w-full h-full object-contain max-w-none"
          style={{ width: containerWidth > 0 ? containerWidth : '100%' }}
        />
      </div>

      {/* Slider Handle */}
      <div 
        className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize shadow-[0_0_10px_rgba(0,0,0,0.5)] z-10"
        style={{ left: `${sliderPosition}%` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center">
          <svg className="w-4 h-4 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <path d="M15 18l-6-6 6-6" />
            <path d="M9 18l6-6-6-6" />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-4 left-4 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur text-xs font-bold text-white/80 border border-white/10 pointer-events-none">
        Original
      </div>
      <div className="absolute top-4 right-4 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur text-xs font-bold text-indigo-400 border border-indigo-500/30 pointer-events-none">
        Upscaled (AI)
      </div>
      
      {label && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur text-xs font-medium text-white pointer-events-none">
            {label}
        </div>
      )}
    </div>
  )
}
