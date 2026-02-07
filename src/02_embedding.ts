import { env, pipeline } from '@huggingface/transformers'
import { readFile, writeFile } from './IOUtils'
env.backends.onnx.logLevel = 'error'

const vectors = []

console.log('Reading chunked corpus...')
const chunkedCorpus = readFile('../corpus/straffeloven_chunks.txt').split('\n')

console.log('Loading embedder model...')
const embedder = await pipeline('feature-extraction', 'Xenova/bge-m3', {
  device: 'auto',
  dtype: 'q8',
})

console.log('Generating embeddings for chunks...')
for (let text of chunkedCorpus) {
  const embedding = (
    await embedder(text, {
      pooling: 'mean',
      normalize: true,
    })
  ).data

  vectors.push({ text, embedding })
}

console.log('Saving embeddings to file...')
writeFile('../corpus/straffeloven_embeddings.json', JSON.stringify(vectors))
