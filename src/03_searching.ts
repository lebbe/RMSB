import { cos_sim, env, pipeline } from '@huggingface/transformers'
import { readFile } from './IOUtils'
env.backends.onnx.logLevel = 'error'

const TRESHOLD = 0.7

const embeddings = JSON.parse(
  readFile('../corpus/straffeloven_embeddings.json'),
).map(({ text, embedding }: { text: string; embedding: any }) => {
  const values = Array.isArray(embedding)
    ? embedding
    : Array.from(embedding instanceof Object ? Object.values(embedding) : [])
  return { text, embedding: values.map(Number) }
})

const embedder = await pipeline('feature-extraction', 'Xenova/bge-m3', {
  device: 'auto',
  dtype: 'q8',
})

const question = 'Hva er straffen for å rane butikken?'

const questionEmbedding = Array.from(
  (
    await embedder(question, {
      pooling: 'mean',
      normalize: true,
    })
  ).data,
)

const results: { text: string; score: number }[] = []

for (let { text, embedding } of embeddings) {
  const score = cos_sim(questionEmbedding, embedding)

  if (score >= TRESHOLD) {
    results.push({ text, score })
  }
}

console.log('\nRelevante tekstutdrag:')
const print = results.sort((a, b) => b.score - a.score).slice(0, 4)
for (let { text, score } of print) {
  console.log(`Troverdighet: ${(score * 100).toFixed(1)}% Tekst: ${text}`)
}
