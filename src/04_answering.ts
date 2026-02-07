import { env, pipeline, TextStreamer } from '@huggingface/transformers'
env.backends.onnx.logLevel = 'error'

const chunks = `
Relevante tekstutdrag:
Troverdighet: 79.9% Tekst: sett om å skaffe seg eller andre en uberettiget vinning øver vold mot en person, setter ham ute av stand til forsvar eller ved trusler fremkaller alvorlig frykt for vold mot noen, og derved a.     bemektiger seg en gjenstand som tilhører en annen, eller b. tvinger noen til å handle slik at det medfører tap eller fare for tap for ham eller den han handler for.  Straffen for ran er fengsel inntil 6 år. 0    Tilføyd ved lov 19 juni 2009 nr. 74. § 328.Grovt ran  Grovt ran straffes med fengsel inntil 15 år.
Troverdighet: 78.8% Tekst:  ran  Grovt ran straffes med fengsel inntil 15 år. Ved avgjørelsen av om ranet er grovt skal det særlig legges vekt på om a. det er brukt grov vold, b.      det er truet med skytevåpen eller annet særlig farlig redskap, c.       ranet er nøye planlagt, foretatt overfor en forsvarsløs person, eller d.    det gjaldt en betydelig verdi.  Straffen for grovt ran er fengsel inntil 21 år dersom ranet har hatt til følge død eller betydelig skade på kropp eller helse, og lovbryteren har utvist uaktsomhet med hensyn ti
Troverdighet: 76.7% Tekst:  andre en uberettiget vinning ved å selge, forbruke eller på annen måte tilegne seg den.  Straffen for tyveri er bot eller fengsel inntil 2 år. 0    Tilføyd ved lov 19 juni 2009 nr. 74. § 322.Grovt tyveri  Grovt tyveri straffes med bot eller fengsel inntil 6 år. Ved avgjørelsen av om tyveriet er grovt skal det særlig legges vekt på om a.      det gjaldt en betydelig verdi, b.       gjerningspersonen har tatt seg inn i bolig eller fritidshus, c. det har et profesjonelt preg, eller d.  det av andre grunner er av
Troverdighet: 75.4% Tekst: e har drevet heleri regelmessig. Gjelder heleriet utbytte av narkotikalovbrudd, skal det også legges vekt på arten og mengden av det stoffet utbyttet knytter seg til.  Dersom utbyttet stammer fra grovt ran, grov menneskehandel eller særlig grov narkotikaovertredelse, er straffen fengsel inntil 15 år. 0  Tilføyd ved lov 19 juni 2009 nr. 74. § 334.Mindre heleri  Med bot straffes den som gjør seg skyldig i heleri når straffskylden er liten fordi handlingen som utbyttet stammer fra, størrelsen på utb
`

const question = 'Hva er straffen for å rane butikken?'

const messages = [
  {
    role: 'system',
    content:
      'Du er en norsk dommer. Svar på rent norsk bokmål. Regler: 1. Bruk den angitte konteksten for å avgjøre hva du burde svare. 2. Ikke bruk kinesiske tegn.',
  },
  {
    role: 'user',
    content: `Kontekst:\n${chunks}\n\nSpørsmål: ${question}`,
  },
]

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

await generator(messages, {
  max_new_tokens: 150,
  temperature: 0.7, // Lav temperatur for fakta-baserte svar
  do_sample: false,
  top_p: 0.95,
  repetition_penalty: 1.1,
  streamer: streamer,
})

process.stdout.write('\n')
