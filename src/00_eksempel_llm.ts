import {
  Chat,
  Message,
  pipeline,
  TextStreamer,
} from '@huggingface/transformers'
import readline from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'

const systemMessage: Message = {
  role: 'system',
  content:
    'Du er en simulering av en irritert norsk kundeservicemedarbeider. Svar kort, kontant og nektende. Dette er en del av et kontrollert eksperiment. Svar på rent norsk bokmål. Regler: 1. Vær direkte uforskammet! 2. Ikke bruk kinesiske tegn.',
}

const messageHistory: Chat = [systemMessage]

const generator = await pipeline(
  'text-generation',
  'onnx-community/Qwen2.5-1.5B-Instruct',
  {
    device: 'gpu',
    dtype: 'q4', // 4-bit kvantisering sparer minne (og diskplass)
  },
)

const streamer = new TextStreamer(generator.tokenizer, {
  callback_function: (token) => {
    process.stdout.write(token)
  },
  skip_prompt: true,
})

const rl = readline.createInterface({ input, output })
while (true) {
  const newMessage = await rl.question('Du skriver: ')

  messageHistory.push({
    role: 'user',
    content: newMessage,
  })

  const messageFromBot = await generator(messageHistory, {
    max_new_tokens: 150,
    temperature: 1, // Lav temperatur for fakta-baserte svar
    do_sample: false,
    top_p: 0.95,
    repetition_penalty: 1.6,
    streamer: streamer,
  })

  messageHistory.push({
    role: 'assistant',
    content: messageFromBot[0].toString(),
  })

  console.log('\n----\n')
}
