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
