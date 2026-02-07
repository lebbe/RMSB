import { cos_sim, env, pipeline } from '@huggingface/transformers'
env.backends.onnx.logLevel = 'error'

const embedder = await pipeline('feature-extraction', 'Xenova/bge-m3', {
  device: 'auto',
  dtype: 'q8',
})

async function getEmbedding(text: string) {
  return (
    await embedder(text, {
      pooling: 'mean',
      normalize: true,
    })
  ).data as number[]
}

const likhet = cos_sim(
  await getEmbedding('Faen det var varmt her'),
  await getEmbedding('Den damen ser virkelig pen ut'),
)

console.log(`Likhet: ${likhet}`)
