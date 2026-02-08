---
marp: true
theme: default
class: invert
paginate: true
---

# RAG med strikk og binders i TypeScript

## Lars-Erik Bruce / Redpill Linpro

---

# Formålet med workshoppen

1. Forstå noen av de grunnleggende byggeklossene i "Chat GPT":
   - Word embedding: Finne "semantikken" i et ord
   - Måle likhet mellom vektorer.
2. Forstå grunnstenene i en RAG.
   - Finne "semantikken" til en frase/setning/chunk.
   - Forhåndsfylle konteksten til LLMen, før den får servert spørsmålet.

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
  await getEmbedding('Man høster som man sår.'),
  await getEmbedding('Som man reder, ligger man.'),
)

console.log(`Likhet: ${likhet}`)
```

---

# Oppgave 0: Embedding

- Lek med eksempelkoden, sjekk semantisk likhet mellom ulike fraser du måtte
  lure på.

- Å laste ned embedding-moddelen kan ta litt tid (ca. 05. GB), så da kan jeg
  snakke litt om embedding og semantikk så lenge!

```bash
npm i
npx tsx src/00_eksempel_embedding.ts
npx tsx src/00_eksempel_llm.ts
```

---

# Ord og semantikk

- sparse vektor: teller opp ord "i nærheten"
- word2vec: lag tette vektorer
- To like vektorer ligger i "samme sted" i det flerdimensjonale rommet
- Moderne teknologier bruker "nevrale nettverk"
- Vi kan måle semantikken i fraser ved å legge sammen vektorene for ord!
- Da må vi huske å normalisere først.

---

# Aritmetikk på semantiske vektorer

Har vi vektoren for "konge", "mann" og "kvinne", så kan vi regne oss frem til vektoren for "dronning".

```
  king: 'the male monarch of a kingdom',
  b: 'a male person',
  c: 'a female person',
  queen: 'the female monarch of a kingdom'
```

```
Likhet mellom "king" og "b":             0.84
Likhet mellom "king" og "c":             0.76
Likhet mellom "b" og "c":                0.89
Likhet mellom "king" og "queen":         0.92

Likhet mellom "king - b + c" og "queen": 0.97
```

---

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<div id="vectorPlot" style="width: 800px; height: 500px; margin: 0 auto;"></div>

<script>
  // 3. Her limer du inn JSON-dataene fra TypeScript-koden din
  const dataPoints = [
  {
    "label": "Konge",
    "x": 0.04,
    "y": -0.02,
    "z": -0.31,
    "color": "#3498db",
    "type": "base"
  },
  {
    "label": "Mann",
    "x": 0.4,
    "y": 0.34,
    "z": -0.08,
    "color": "#95a5a6",
    "type": "base"
  },
  {
    "label": "Kvinne",
    "x": 0.38,
    "y": 0.06,
    "z": 0.29,
    "color": "#95a5a6",
    "type": "base"
  },
  {
    "label": "Dronning",
    "x": 0.02,
    "y": -0.24,
    "z": -0.04,
    "color": "#f1c40f",
    "type": "target"
  },
  {
    "label": "Resultat",
    "x": 0.01,
    "y": -0.31,
    "z": 0.06,
    "color": "#2ecc71",
    "type": "calculated"
  },
  {
    "label": "Konge - Mann",
    "x": -0.86,
    "y": 0.18,
    "z": 0.07,
    "color": "#e67e22",
    "type": "calculated"
  }
];

  const trace = {
    x: dataPoints.map(p => p.x),
    y: dataPoints.map(p => p.y),
    z: dataPoints.map(p => p.z),
    mode: 'markers+text',
    type: 'scatter3d',
    text: dataPoints.map(p => p.label),
    textposition: 'top center',
    marker: { size: 8, color: dataPoints.map(p => p.color), opacity: 0.8 }
  };

  const layout = {
    margin: {l: 0, r: 0, b: 0, t: 0},
    scene: {
      xaxis: {title: 'PCA 1'},
      yaxis: {title: 'PCA 2'},
      zaxis: {title: 'PCA 3'}
    }
  };

  // Skjul GUI-knapper og toolbar
  const plotlyConfig = { displayModeBar: false };
  Plotly.newPlot('vectorPlot', [trace], layout, plotlyConfig);
</script>

---

# RAG: Pipeline

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

# Andre morsomme ting å bruke embedding til

- Semantisk caching: I stedet for å generere det samme svaret på nytt igjen,
  for et "semantisk" likt spørsmål, hent opp svaret fra en semantisk cache.
- ***

# Oppgave 04: Answering

Her har jeg allerede lagt opp en LLM som svarer på spørsmål. Men klipp inn
faktiske relevante chunks inn i `chunks`, og skriv faktisk spørsmål som du
lurer på inn i `question`.

---

# Bonusoppgave 05: RAG

Lag et terminalprogram som tar i mot et spørsmål fra bruker, og
som svarer brukeren ved hjelp av alle verktøy vi har bygd så langt.

---

# Going prod

- **@pinecone-database/pinecone**: Vektordatabase for prod.
- **Vertex AI Search**: Fullstendig RAG-pipeline i Google Cloud.
