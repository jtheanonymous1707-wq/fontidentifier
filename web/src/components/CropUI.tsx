'use client'
import { useState, useRef, useCallback } from 'react'
import ReactCrop, { Crop, PixelCrop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'

type Props = {
  imageSrc: string
  onCropConfirmed: (croppedBlob: Blob) => void
  onReset: () => void
}

export default function CropUI({ imageSrc, onCropConfirmed, onReset }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [crop, setCrop] = useState<Crop>({
    unit: '%',
    x: 10, y: 25,
    width: 80, height: 50
  })
  const [completedCrop, setCompletedCrop] = useState<PixelCrop>()

  const getCroppedBlob = useCallback((): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      const img   = imgRef.current
      const c     = completedCrop
      if (!img || !c) return reject('No crop selected')

      const canvas = document.createElement('canvas')
      const scaleX = img.naturalWidth  / img.width
      const scaleY = img.naturalHeight / img.height

      canvas.width  = c.width
      canvas.height = c.height

      const ctx = canvas.getContext('2d')
      if (!ctx) return reject('Canvas context unavailable')

      ctx.drawImage(
        img,
        c.x * scaleX, c.y * scaleY,
        c.width * scaleX, c.height * scaleY,
        0, 0,
        c.width, c.height
      )

      canvas.toBlob((blob) => {
        if (blob) resolve(blob)
        else reject('Failed to create blob')
      }, 'image/png')
    })
  }, [completedCrop])

  const handleConfirm = async () => {
    try {
      const blob = await getCroppedBlob()
      onCropConfirmed(blob)
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Instruction banner */}
      <div className="w-full bg-blue-50 border border-blue-200 rounded-xl px-4 py-3 text-blue-700 text-sm text-center">
        ✂️ Draw a box around <strong>one line of text</strong> — avoid backgrounds and decorations
      </div>

      <ReactCrop
        crop={crop}
        onChange={(c) => setCrop(c)}
        onComplete={(c) => setCompletedCrop(c)}
        minWidth={40}
        minHeight={20}
        className="max-w-full rounded-xl shadow-lg overflow-hidden"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Upload to crop"
          className="max-w-full max-h-[60vh] object-contain"
        />
      </ReactCrop>

      <div className="flex gap-3">
        <button
          onClick={onReset}
          className="px-6 py-2.5 rounded-xl border border-gray-300 text-gray-600
                     hover:bg-gray-100 transition-colors font-medium"
        >
          ← Upload different image
        </button>
        <button
          onClick={handleConfirm}
          disabled={!completedCrop}
          className="px-8 py-2.5 rounded-xl bg-blue-600 text-white font-semibold
                     hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed
                     transition-colors shadow-sm"
        >
          Identify Font →
        </button>
      </div>
    </div>
  )
}
