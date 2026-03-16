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
