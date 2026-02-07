import { env, pipeline } from '@huggingface/transformers'
import { readFile, writeFile } from './IOUtils'
env.backends.onnx.logLevel = 'error'

const vectors = [] as { text: string; embedding: number[] }[]

console.log('Reading chunked corpus...')
const chunkedCorpus = readFile('../corpus/straffeloven_chunks.txt').split('\n')

console.log('Loading embedder model...')

// TODO IMPLEMENT ME

console.log('Generating embeddings for chunks...')

// TODO IMPLEMENT ME

console.log('Saving embeddings to file...')
writeFile('../corpus/straffeloven_embeddings.json', JSON.stringify(vectors))
