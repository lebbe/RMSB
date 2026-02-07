import { cos_sim, env, pipeline, TextStreamer } from '@huggingface/transformers'
import { createInterface } from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'
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
const embedder = await pipeline('feature-extraction', 'Xenova/bge-m3', {
  device: 'auto',
  dtype: 'q8',
})

console.log('Loading generator model...')
const generator = await pipeline(
  'text-generation',
  'onnx-community/Qwen2.5-1.5B-Instruct',
  {
    device: 'gpu',
    dtype: 'q4', // 4-bit saves memory
  },
)

const rl = createInterface({ input, output })

while (true) {
  // # 1 Prompt user for a question
  const question = (await rl.question('User: ')).trim()

  if (question.toLowerCase() === 'exit') {
    rl.close()
    break
  }

  // # 2 Generate embedding for the question
  const questionEmbedding = (
    await embedder(question, {
      pooling: 'mean',
      normalize: true,
    })
  ).data as number[]

  // # 3 Search for relevant chunks based on cosine similarity
  const results = []
  for (let { text, embedding } of embeddings) {
    const score = cos_sim(questionEmbedding, embedding)

    if (score >= TRESHOLD) {
      results.push({ text, score })
    }
  }

  // # 4 Format the retrieved chunks as context for the LLM
  const context = results
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
    .map(
      ({ text, score }) =>
        `Troverdighet: ${(score * 100).toFixed(1)}% Tekst: ${text}`,
    )
    .join('\n')

  const messages = [
    {
      role: 'system',
      content:
        'Du er en norsk dommer. Svar på rent norsk bokmål. Regler: 1. Bruk den angitte konteksten for å avgjøre hva du burde svare. 2. Ikke bruk kinesiske tegn.',
    },
    {
      role: 'user',
      content: `Kontekst:\n${context}\n\nSpørsmål: ${question}`,
    },
  ]

  // # 5 Generate an answer using the LLM with the retrieved context

  process.stdout.write('System: ')
  const streamer = new TextStreamer(generator.tokenizer, {
    callback_function: (token) => {
      process.stdout.write(token)
    },
    skip_prompt: true,
  })

  await generator(messages, {
    max_new_tokens: 150,
    temperature: 0.7, // Low temperature for factual answers
    do_sample: false,
    top_p: 0.95,
    repetition_penalty: 1.1,
    streamer: streamer,
  })

  process.stdout.write('\n')
}
