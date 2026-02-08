import { cos_sim, pipeline } from '@huggingface/transformers'
import { PCA } from 'ml-pca'
import fs from 'fs'

// --- KONFIGURASJON ---
const MODEL_NAME = 'Xenova/multilingual-e5-large' // Eller 'intfloat/multilingual-e5-large'
const OUTPUT_FILE = 'dataPoints.json'

// Ordene vi skal jobbe med (Legg merke til "query:" prefixet som er kritisk for E5)
const TERMS = {
  a: 'query: the male monarch of a kingdom',
  b: 'query: a male person',
  c: 'query: a female person',
  target: 'query: the female monarch of a kingdom', // Fasiten
}

// --- TYPE DEFINISJONER ---
interface PlotlyPoint {
  label: string
  x: number
  y: number
  z: number
  color: string
  type: 'base' | 'target' | 'calculated'
}

// --- VEKTOR MATTE ---
const vecMath = {
  add: (a: number[], b: number[]) => a.map((v, i) => v + b[i]),
  sub: (a: number[], b: number[]) => a.map((v, i) => v - b[i]),
  dot: (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0),
  mag: (a: number[]) => Math.sqrt(a.reduce((sum, v) => sum + v * v, 0)),
  cosine: (a: number[], b: number[]) => {
    return vecMath.dot(a, b) / (vecMath.mag(a) * vecMath.mag(b))
  },
}

async function main() {
  console.log(`Laster modell: ${MODEL_NAME}...`)
  const extractor = await pipeline('feature-extraction', MODEL_NAME, {
    dtype: 'q8', // Kvantifisert for hastighet
  })

  // Hjelpefunksjon for embedding
  const getVec = async (text: string): Promise<number[]> => {
    const out = await extractor(text, { pooling: 'mean', normalize: true })
    return Array.from(out.data) as number[]
  }

  console.log('Genererer embeddings...')
  const vecKonge = await getVec(TERMS.a) // Konge
  const vecMann = await getVec(TERMS.b) // Mann
  const vecKvinne = await getVec(TERMS.c) // Kvinne
  const vecTarget = await getVec(TERMS.target) // Dronning

  console.log('Utfører aritmetikk: Konge - Mann + Kvinne...')
  // 1. Trekk fra Mann (fjerner maskuline trekk)
  const kongeMinusMann = vecMath.sub(vecKonge, vecMann)
  // 2. Legg til Kvinne (legger til feminine trekk)
  const kongeMinusMannPlussKvinne = vecMath.add(kongeMinusMann, vecKvinne)

  // Sjekk likhet før PCA (for å verifisere at matten fungerer)
  const sim = vecMath.cosine(kongeMinusMannPlussKvinne, vecTarget)
  console.log(`Cosinus-likhet mellom BEREGNET og DRONNING: ${sim.toFixed(4)}`)

  // --- PCA (3D) ---
  console.log('Kjører PCA for å redusere til 3 dimensjoner...')

  // Matrise: Rekkefølgen her bestemmer indeksene i coords
  const matrix = [
    vecKonge,
    vecMann,
    vecKvinne,
    vecTarget,
    kongeMinusMannPlussKvinne,
    kongeMinusMann,
  ]

  const pca = new PCA(matrix)
  const coords = pca.predict(matrix, { nComponents: 3 }).to2DArray()

  // --- BYGG DATA FOR PLOTLY ---

  // Hjelpefunksjon for avrunding
  function round2(n: number) {
    return Math.round(n * 100) / 100
  }

  const dataPoints: PlotlyPoint[] = [
    {
      label: 'Konge',
      x: round2(coords[0][0]),
      y: round2(coords[0][1]),
      z: round2(coords[0][2]),
      color: '#3498db',
      type: 'base',
    },
    {
      label: 'Mann',
      x: round2(coords[1][0]),
      y: round2(coords[1][1]),
      z: round2(coords[1][2]),
      color: '#95a5a6',
      type: 'base',
    },
    {
      label: 'Kvinne',
      x: round2(coords[2][0]),
      y: round2(coords[2][1]),
      z: round2(coords[2][2]),
      color: '#95a5a6',
      type: 'base',
    },
    {
      label: 'Dronning',
      x: round2(coords[3][0]),
      y: round2(coords[3][1]),
      z: round2(coords[3][2]),
      color: '#f1c40f',
      type: 'target',
    }, // Gull
    {
      label: 'Resultat',
      x: round2(coords[4][0]),
      y: round2(coords[4][1]),
      z: round2(coords[4][2]),
      color: '#2ecc71',
      type: 'calculated',
    },
    {
      label: 'Konge - Mann',
      x: round2(coords[5][0]),
      y: round2(coords[5][1]),
      z: round2(coords[5][2]),
      color: '#e67e22',
      type: 'calculated',
    },
  ]

  // Lagre til fil eller print
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(dataPoints, null, 2))
  console.log(`\nFerdig! Data lagret til ${OUTPUT_FILE}`)
  console.log('Kopier innholdet i denne filen inn i HTML-koden din.')

  console.log(
    `Likhet mellom "konge" og "mann": ${round2(cos_sim(vecKonge, vecMann))}`,
  )
  console.log(
    `Likhet mellom "konge" og "kvinne": ${round2(cos_sim(vecKonge, vecKvinne))}`,
  )
  console.log(
    `Likhet mellom "mann" og "kvinne": ${round2(cos_sim(vecMann, vecKvinne))}`,
  )
  console.log(
    `Likhet mellom "konge" og "dronning": ${round2(cos_sim(vecKonge, vecTarget))}`,
  )
  console.log(
    `Likhet mellom "konge - mann" og "dronning": ${round2(cos_sim(kongeMinusMann, vecTarget))}`,
  )
  console.log(
    `Likhet mellom "konge - mann + kvinne" og "dronning": ${round2(cos_sim(kongeMinusMannPlussKvinne, vecTarget))}`,
  )
}

main().catch(console.error)
