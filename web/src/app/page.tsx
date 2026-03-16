'use client'
import { useState, useEffect, useRef } from 'react'
import Uploader  from '@/components/Uploader'
import CropUI    from '@/components/CropUI'
import FontResult from '@/components/FontResult'
import { FontMatch } from '@/lib/types'

type Stage = 'upload' | 'crop' | 'loading' | 'results' | 'error'

const HF_API_URL = process.env.NEXT_PUBLIC_HF_API_URL!

export default function HomePage() {
  const [stage,    setStage]    = useState<Stage>('upload')
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [matches,  setMatches]  = useState<FontMatch[]>([])
  const [errMsg,   setErrMsg]   = useState<string>('')
  const [warming,  setWarming]  = useState(true)
  const croppedBlobRef = useRef<Blob | null>(null)

  // Ping HF Space on page load to wake it up
  useEffect(() => {
    fetch(`${HF_API_URL}/ping`)
      .then(() => setWarming(false))
      .catch(() => setWarming(false))
  }, [])

  const handleImageSelected = (dataUrl: string) => {
    setImageSrc(dataUrl)
    setStage('crop')
  }

  const handleCropConfirmed = async (blob: Blob) => {
    croppedBlobRef.current = blob
    setStage('loading')

    try {
      const form = new FormData()
      form.append('file', blob, 'crop.png')

      const res  = await fetch('/api/identify', { method: 'POST', body: form })
      const data = await res.json()

      if (!res.ok || data.error) {
        setErrMsg(data.error ?? 'Something went wrong')
        setStage('error')
        return
      }

      setMatches(data.matches)
      setStage('results')
    } catch {
      setErrMsg('Network error — please try again')
      setStage('error')
    }
  }

  const handleFeedback = async (fontId: number, isCorrect: boolean) => {
    const blob = croppedBlobRef.current
    if (!blob) return

    // Upload cropped image to Supabase Storage for feedback record
    // Note: In a production app, you'd upload the blob to storage first
    // For now, we'll use a placeholder URL as per the schema requirement
    const imageUrl = `feedback/${Date.now()}.png`

    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_url:         imageUrl,
        predicted_font_id: matches[0]?.id,
        correct_font_id:   isCorrect ? fontId : undefined,
        is_correct:        isCorrect,
      }),
    })
  }

  const reset = () => {
    setStage('upload')
    setImageSrc(null)
    setMatches([])
    setErrMsg('')
    croppedBlobRef.current = null
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-3xl mx-auto px-4 py-12">

        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-black text-gray-900 tracking-tight mb-3">
            Font Identifier
          </h1>
          <p className="text-gray-500 text-lg">
            Upload any image, crop the text, and identify the font instantly.
          </p>
          {warming && (
            <p className="text-blue-500 text-sm mt-2 animate-pulse">
              ⚡ Warming up model...
            </p>
          )}
        </div>

        {/* Stage: Upload */}
        {stage === 'upload' && (
          <Uploader onImageSelected={handleImageSelected} />
        )}

        {/* Stage: Crop */}
        {stage === 'crop' && imageSrc && (
          <CropUI
            imageSrc={imageSrc}
            onCropConfirmed={handleCropConfirmed}
            onReset={reset}
          />
        )}

        {/* Stage: Loading */}
        {stage === 'loading' && (
          <div className="flex flex-col items-center justify-center py-32 gap-4">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent
                            rounded-full animate-spin" />
            <p className="text-gray-500 font-medium">Identifying font...</p>
          </div>
        )}

        {/* Stage: Results */}
        {stage === 'results' && (
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-xl font-bold text-gray-800">
                Top {matches.length} matches
              </h2>
              <button
                onClick={reset}
                className="text-sm text-blue-600 hover:underline font-medium"
              >
                Try another image
              </button>
            </div>

            {matches.map((match, i) => (
              <FontResult
                key={match.id}
                match={match}
                rank={i + 1}
                imageUrl={`feedback/${Date.now()}.png`}
                onFeedback={handleFeedback}
              />
            ))}
          </div>
        )}

        {/* Stage: Error */}
        {stage === 'error' && (
          <div className="flex flex-col items-center gap-4 py-24 text-center">
            <div className="text-5xl">⚠️</div>
            <p className="text-gray-700 font-medium">{errMsg}</p>
            <button
              onClick={reset}
              className="px-6 py-2.5 rounded-xl bg-gray-900 text-white
                         hover:bg-gray-700 transition-colors font-medium"
            >
              Try again
            </button>
          </div>
        )}

        {/* Footer */}
        <p className="text-center text-gray-400 text-xs mt-16">
          Built with Next.js · Supabase · PyTorch · Hugging Face Spaces
        </p>
      </div>
    </main>
  )
}
