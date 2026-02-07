import { readFile, writeFile } from './IOUtils'

const CHUNK_SIZE = 500
const OVERLAP = 50
const STEP = CHUNK_SIZE - OVERLAP

const txt = readFile('../corpus/straffeloven.txt')

const chunks = []
for (let start = 0; start < txt.length; start += STEP) {
  const chunk = txt.slice(start, start + CHUNK_SIZE)
  if (chunk.length === 0) break
  chunks.push(chunk)
}

writeFile('../corpus/straffeloven_chunks.txt', chunks.join('\n'))
