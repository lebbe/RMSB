import { pipeline } from '@huggingface/transformers'
import { PCA } from 'ml-pca'
import fs from 'fs'

// --- KONFIGURASJON ---
const MODEL_NAME = 'Xenova/multilingual-e5-large'
const OUTPUT_FILE = 'dataPoints_words.json'

// Substantiver og verb (engelsk, med "query:" prefix)
const NOUNS = [
  'query: apple',
  'query: banana',
  'query: car',
  'query: house',
  'query: dog',
  'query: cat',
  'query: friendship',
  'query: sadness',
  'query: computer',
  'query: happiness',
]

const VERBS = [
  'query: run',
  'query: jump',
  'query: eat',
  'query: sleep',
  'query: drive',
  'query: swim',
  'query: think',
  'query: laugh',
  'query: cry',
  'query: build',
]

// --- TYPE DEFINISJONER ---
interface PlotlyPoint {
  label: string
  x: number
  y: number
  z: number
  color: string
  type: 'noun' | 'verb'
}

async function main() {
  console.log(`Laster modell: ${MODEL_NAME}...`)
  const extractor = await pipeline('feature-extraction', MODEL_NAME, {
    dtype: 'q8',
  })

  // Embedding for alle ord
  const nounVecs = await Promise.all(
    NOUNS.map((text) =>
      extractor(text, { pooling: 'mean', normalize: true }).then(
        (out) => Array.from(out.data) as number[],
      ),
    ),
  )
  const verbVecs = await Promise.all(
    VERBS.map((text) =>
      extractor(text, { pooling: 'mean', normalize: true }).then(
        (out) => Array.from(out.data) as number[],
      ),
    ),
  )

  // PCA for å redusere til 3D
  const matrix = [...nounVecs, ...verbVecs]
  const pca = new PCA(matrix)
  const coords = pca.predict(matrix, { nComponents: 3 }).to2DArray()

  // Hjelpefunksjon for avrunding
  function round2(n: number) {
    return Math.round(n * 100) / 100
  }

  // Bygg dataPoints
  const dataPoints: PlotlyPoint[] = [
    ...NOUNS.map((label, i) => ({
      label: label.replace('query: ', ''),
      x: round2(coords[i][0]),
      y: round2(coords[i][1]),
      z: round2(coords[i][2]),
      color: '#3498db',
      type: 'noun' as const,
    })),
    ...VERBS.map((label, i) => ({
      label: label.replace('query: ', ''),
      x: round2(coords[NOUNS.length + i][0]),
      y: round2(coords[NOUNS.length + i][1]),
      z: round2(coords[NOUNS.length + i][2]),
      color: '#e67e22',
      type: 'verb' as const,
    })),
  ]

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(dataPoints, null, 2))
  console.log(`\nFerdig! Data lagret til ${OUTPUT_FILE}`)
  console.log('Kopier innholdet i denne filen inn i HTML-koden din.')
}

main().catch(console.error)
