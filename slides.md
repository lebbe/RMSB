---
marp: true
theme: default
class: invert
paginate: true
---

# RAG med strikk og binders i TypeScript

## Lars-Erik Bruce / Redpill Linpro

---

# RAG: Retrieval Augmented Generation

Vi hjelper en LLM til å komme med "riktig" svar ved å legge "relevante" deler av dokumenter (chunks) inn i konteksten, før den begynner å generere et svar.

---

# Embeddingsmodeller

- Input: Diskrete data som setninger eller bilder.
- Output: En vektor med flyttall, som representerer semantisk mening.
- (Egentlig: En vektor per ord/token, som blir normalisert.)
- To setninger som betyr omtrent det samme, får ca samme vektor.

## "Det var en gang et menneske."

[0.123213, 0.7563456, 0.3452345, 0.32453245, -0.873784, 0.000123, ...]

---

# Kode-eksempel (ligger vedlagt)

```TypeScript
import { cos_sim, env, pipeline } from '@huggingface/transformers'
env.backends.onnx.logLevel = 'error' // Slå av teit logging

const embedder = await pipeline('feature-extraction', 'Xenova/bge-m3', {
  device: 'auto',
  dtype: 'q8', // Last ned mye mindre modell
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
```

---

# Oppgave 0: Embedding

Lek med eksempelkoden, sjekk semantisk likhet mellom ulike fraser du måtte
lure på.

---

# Mer om RAG

asdf

---

# Pipeline

0. Skaff et korpus vi kan bruke som kilde.
1. Del denne opp i chunks.
2. Lag en semantisk vektor for hver chunk.
3. Finn chunks som er semantisk mest lik et spørsmål.
4. Få en LLM til å svare på spørsmålet med utgangspunkt i chunks.

---

# Oppgave 01: Chunking

1. Finn et større dokument du vil bruke som kilde.
2. Lagre denne som en .txt fil i `corpus`
3. Lag chunks av denne ved å fullføre koden i `01_chunking.ts`

---

# Oppgave 02: Embedding

Skriv et program som går gjennom hver chunk, beregner en vektor per chunk, og
lagrer denne sammen med chunk-teksten i en JSON-fil.

```TypeScript
[
  {
    text: string
    embedding: number[]
  },
  ...
]
```

---

# Oppgave 03: Searching

1. Les inn alle embeddings lagd i Oppgave 4.
2. Lag en embedding for et spørsmål `question`.
3. Mål kosinus-likhet mellom spørsmålet og alle chunks med `cos_sim`.
4. `console.log` ut de chunks som "ligner" mest på spørsmålet.

---

# Oppgave 04: Answering

Her har jeg allerede lagt opp en LLM som svarer på spørsmål. Men klipp inn
faktiske relevante chunks inn i `chunks`, og skriv faktisk spørsmål som du
lurer på inn i `question`.

---

# Bonusoppgave 05: RAG

Lag et terminalprogram som tar i mot et spørsmål fra bruker, og
som svarer brukeren ved hjelp av alle verktøy vi har bygd så langt.
