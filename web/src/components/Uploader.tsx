'use client'
import { useRef, useState } from 'react'

type Props = {
  onImageSelected: (dataUrl: string) => void
}

export default function Uploader({ onImageSelected }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => onImageSelected(reader.result as string)
    reader.readAsDataURL(file)
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) handleFile(file)
      }}
      className={`
        w-full border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer
        transition-all duration-200
        ${dragging
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
        }}
      />
      <div className="text-5xl mb-4">🖼️</div>
      <p className="text-gray-600 font-medium text-lg">
        Drop an image here or click to upload
      </p>
      <p className="text-gray-400 text-sm mt-2">
        JPEG, PNG, WebP · Max 5MB
      </p>
    </div>
  )
}
