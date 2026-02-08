import { cos_sim, env, pipeline } from '@huggingface/transformers'
env.backends.onnx.logLevel = 'error'

console.log(
  'Laster embedding-modellen. Dette kan ta litt tid første gang, så ta en kaffe eller noe.',
)
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
  await getEmbedding('Man høster som man sår.'),
  await getEmbedding('Som man reder, ligger man.'),
)

console.log(`Likhet: ${likhet}`)
