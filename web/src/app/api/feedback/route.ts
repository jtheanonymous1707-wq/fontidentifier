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

    await supabase.from('feedback').insert({
      image_url,
      predicted_font_id,
      correct_font_id: correct_font_id ?? null,
      is_correct,
    })

    return NextResponse.json({ status: 'recorded' })
  } catch (err) {
    console.error('Feedback error:', err)
    return NextResponse.json({ error: 'Failed to save feedback' }, { status: 500 })
  }
}
