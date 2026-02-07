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

console.log('Loading embedding model...')

// TODO IMPLEMENT ME

const question = 'Hva er straffen for å rane butikken?'

console.log('Generating embedding for the question...')

// TODO IMPLEMENT ME

console.log('Searching for relevant chunks based on cosine similarity...')

const results: { text: string; score: number }[] = []

// TODO IMPLEMENT ME

console.log('\nRelevante tekstutdrag:')

// TODO IMPLEMENT ME
