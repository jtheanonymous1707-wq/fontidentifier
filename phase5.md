<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Continue phase 5

Here's the full Phase 5 implementation guide.

***

## Phase 5: Frontend (Next.js on Vercel)

### Final Folder Structure

```
web/src/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                    ← main page
│   └── api/
│       ├── identify/
│       │   └── route.ts            ← calls HF Space + Supabase
│       └── feedback/
│           └── route.ts            ← stores user corrections
├── components/
│   ├── Uploader.tsx                ← drag & drop upload
│   ├── CropUI.tsx                  ← crop selection
│   └── FontResult.tsx              ← result cards
└── lib/
    ├── supabase.ts                 ← already done Phase 1
    └── types.ts                    ← shared TypeScript types
```


***

### Step 1 — Install Dependencies

```bash
cd web
npm install react-image-crop
npm install @radix-ui/react-progress
```


***

### Step 2 — Define Shared Types

Create `web/src/lib/types.ts`:

```typescript
export type FontMatch = {
  id: number
  name: string
  category: string
  google_fonts_url: string | null
  preview_url: string | null
  similarity: number
  confidence: number | null
}

export type IdentifyResponse = {
  matches: FontMatch[]
  error?: string
}

export type FeedbackPayload = {
  image_url: string
  predicted_font_id: number
  correct_font_id?: number
  is_correct: boolean
}
```


***

### Step 3 — API Route: `/api/identify`

Create `web/src/app/api/identify/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'

const HF_API_URL = process.env.NEXT_PUBLIC_HF_API_URL

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const file = formData.get('file') as File | null

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    // Validate file type
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      return NextResponse.json({ error: 'Unsupported file type. Use JPEG, PNG or WebP.' }, { status: 415 })
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      return NextResponse.json({ error: 'File too large. Max 5MB.' }, { status: 413 })
    }

    // Forward to Hugging Face Space
    const hfForm = new FormData()
    hfForm.append('file', file)

    const hfRes = await fetch(`${HF_API_URL}/identify`, {
      method: 'POST',
      body: hfForm,
    })

    if (!hfRes.ok) {
      const err = await hfRes.text()
      return NextResponse.json({ error: `Model error: ${err}` }, { status: 500 })
    }

    const data = await hfRes.json()
    return NextResponse.json(data)

  } catch (err) {
    console.error('Identify error:', err)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
```


***

### Step 4 — API Route: `/api/feedback`

Create `web/src/app/api/feedback/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

// Use service role key on server side only
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_KEY!
)

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { image_url, predicted_font_id, correct_font_id, is_correct } = body

    if (!image_url || predicted_font_id === undefined || is_correct === undefined) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 422 })
    }

    await supabase.table('feedback').insert({
      image_url,
      predicted_font_id,
      correct_font_id: correct_font_id ?? null,
      is_correct,
    })

    return NextResponse.json({ status: 'recorded' })
  } catch (err) {
    return NextResponse.json({ error: 'Failed to save feedback' }, { status: 500 })
  }
}
```

Add `SUPABASE_SERVICE_KEY` to your `.env.local`:

```env
SUPABASE_SERVICE_KEY=your-service-role-key   # server only, no NEXT_PUBLIC prefix
```


***

### Step 5 — Uploader Component

Create `web/src/components/Uploader.tsx`:

```typescript
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
        const file = e.dataTransfer.files[^0]
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
          const file = e.target.files?.[^0]
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
```


***

### Step 6 — CropUI Component

Create `web/src/components/CropUI.tsx`:[^1]

```typescript
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
```


***

### Step 7 — FontResult Component

Create `web/src/components/FontResult.tsx`:

```typescript
'use client'
import { useState } from 'react'
import { FontMatch } from '@/lib/types'

type Props = {
  match: FontMatch
  rank: number
  imageUrl: string
  onFeedback: (fontId: number, isCorrect: boolean) => void
}

const CATEGORY_COLORS: Record<string, string> = {
  'sans-serif' : 'bg-blue-100   text-blue-700',
  'serif'      : 'bg-purple-100 text-purple-700',
  'monospace'  : 'bg-green-100  text-green-700',
  'display'    : 'bg-orange-100 text-orange-700',
  'handwriting': 'bg-pink-100   text-pink-700',
}

export default function FontResult({ match, rank, imageUrl, onFeedback }: Props) {
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

  const handleFeedback = (isCorrect: boolean) => {
    setFeedback(isCorrect ? 'up' : 'down')
    onFeedback(match.id, isCorrect)
  }

  const categoryColor = CATEGORY_COLORS[match.category] ?? 'bg-gray-100 text-gray-600'

  return (
    <div className={`
      relative flex items-center justify-between gap-4 p-4 rounded-2xl border
      transition-all duration-200
      ${rank === 1
        ? 'border-blue-300 bg-blue-50 shadow-md'
        : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'}
    `}>
      {/* Rank badge */}
      <div className={`
        flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
        font-bold text-sm
        ${rank === 1 ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-500'}
      `}>
        {rank}
      </div>

      {/* Font info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-gray-900 text-lg truncate">
            {match.name}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${categoryColor}`}>
            {match.category}
          </span>
          {rank === 1 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-600 text-white font-medium">
              Best match
            </span>
          )}
        </div>

        {/* Similarity bar */}
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500
                ${rank === 1 ? 'bg-blue-500' : 'bg-gray-400'}`}
              style={{ width: `${match.similarity}%` }}
            />
          </div>
          <span className="text-xs text-gray-500 flex-shrink-0">
            {match.similarity.toFixed(1)}% match
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {match.google_fonts_url && (
          <a
            href={match.google_fonts_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs px-3 py-1.5 rounded-lg bg-gray-900 text-white
                       hover:bg-gray-700 transition-colors font-medium"
          >
            View font ↗
          </a>
        )}

        {/* Thumbs feedback */}
        <div className="flex gap-1">
          <button
            onClick={() => handleFeedback(true)}
            disabled={feedback !== null}
            title="This is correct"
            className={`p-1.5 rounded-lg transition-colors text-base
              ${feedback === 'up'
                ? 'bg-green-100 text-green-600'
                : 'hover:bg-gray-100 text-gray-400 disabled:opacity-30'}`}
          >
            👍
          </button>
          <button
            onClick={() => handleFeedback(false)}
            disabled={feedback !== null}
            title="This is wrong"
            className={`p-1.5 rounded-lg transition-colors text-base
              ${feedback === 'down'
                ? 'bg-red-100 text-red-500'
                : 'hover:bg-gray-100 text-gray-400 disabled:opacity-30'}`}
          >
            👎
          </button>
        </div>
      </div>
    </div>
  )
}
```


***

### Step 8 — Main Page

Create `web/src/app/page.tsx`:

```typescript
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
    const imageUrl = `feedback/${Date.now()}.png`

    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_url:         imageUrl,
        predicted_font_id: matches[^0]?.id,
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
```


***

### Step 9 — Update `layout.tsx`

Replace `web/src/app/layout.tsx`:

```typescript
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Font Identifier — Identify any font from an image',
  description: 'Upload an image, crop the text, and instantly identify the font using AI.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
      </body>
    </html>
  )
}
```


***

### Step 10 — Test Locally

```bash
cd web
npm run dev
```

Visit `http://localhost:3000` and walk through the full flow:

- Upload an image ✅
- Crop a text region ✅
- See "Identifying font..." spinner ✅
- See top 5 results with similarity bars ✅
- Click 👍/👎 on results ✅
- Click "View font ↗" to open Google Fonts ✅

***

### Step 11 — Commit and Deploy to Vercel

```bash
cd web
git add .
git commit -m "Phase 5: complete frontend — uploader, crop UI, font results, feedback"
git push origin main
```

Then:

1. Go to [vercel.com](https://vercel.com) → **New Project** → import your `font-identifier` repo
2. Set **Root Directory** to `web`
3. Add all environment variables from `.env.local` in the Vercel dashboard
4. Click **Deploy** — live in ~2 minutes

***

## Phase 5 Checklist

- [ ] `types.ts` created with `FontMatch`, `IdentifyResponse`, `FeedbackPayload`
- [ ] `/api/identify/route.ts` forwards to HF Space and returns matches
- [ ] `/api/feedback/route.ts` stores corrections in Supabase
- [ ] `Uploader.tsx` handles drag \& drop + file input
- [ ] `CropUI.tsx` renders crop selection with canvas extraction
- [ ] `FontResult.tsx` shows rank, similarity bar, Google Fonts link, thumbs feedback
- [ ] `page.tsx` manages all 5 stages: upload → crop → loading → results → error
- [ ] HF Space warmup ping fires on page load
- [ ] `SUPABASE_SERVICE_KEY` added to `.env.local` (server only)
- [ ] Local dev test passes full flow end to end
- [ ] Deployed to Vercel with all env vars set

You're now ready for **Phase 6: Deploy \& Polish** — adding the portfolio extras, custom domain, and the "How it works" section.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.npmjs.com/package/react-image-crop

[^2]: https://www.youtube.com/watch?v=odscV57kToU

[^3]: https://www.youtube.com/watch?v=_rkx75HWU_0

[^4]: https://forum.freecodecamp.org/t/i-need-help-in-image-cropping-plugin-react-image-crop/688194

[^5]: https://snyk.io/advisor/npm-package/react-image-crop/example

[^6]: https://docs.uploadthing.com/getting-started/appdir

[^7]: https://www.npmjs.com/package/react-easy-crop

[^8]: https://blog.logrocket.com/handling-file-uploads-next-js-using-uploadthing/

[^9]: https://img.ly/blog/how-to-crop-an-image-in-react-with-react-image-crop/

[^10]: https://geekyants.com/blog/uploading-folders-and-files-to-shared-drives-using-nextjs-api-route-handlers

[^11]: https://stackoverflow.com/questions/75170252/how-do-i-display-the-cropped-part-using-react-image-crop-it-appears-as-a-black

[^12]: https://github.com/vercel/next.js/discussions/50165

[^13]: https://stackoverflow.com/questions/52273880/get-cropped-image-via-react-image-crop-module

[^14]: https://www.reddit.com/r/nextjs/comments/13ouqbx/file_upload_with_new_route_handler_at_app/

[^15]: https://www.geeksforgeeks.org/reactjs/how-to-crop-images-in-reactjs/

