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
