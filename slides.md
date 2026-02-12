---
marp: true
theme: default
class: invert
paginate: true
math: mathjax
author: Lars-Erik Bruce
---

# RAG med strikk og binders i TypeScript

## Lars-Erik Bruce / Redpill Linpro

---

# Formålet med workshoppen

1. Forstå én av de grunnleggende byggeklossene i store språkmodeller:
   - Word embedding: Finne "semantikken" (meningen) i et ord
   - Vektorsemantikk: Måle (semantisk) likhet mellom vektorer.
2. Ha kunnskapen som skal til for å lage egen RAG.
   - Finne "semantikken" til et innsendt spørsmål (prompt).
   - Slå opp chunks/fraser/dokumenter med lignende "mening"
   - Forhåndsfylle konteksten til LLMen med disse, før den får servert spørsmålet.

---

# Om Lars-Erik Bruce

Skrevet en masteroppgave innen språkteknologier: "Ontology-driven information extraction and structuring in the clinical domain".

Jobber nå, blant annet, med å fasilitere agentisk AI på kundeservice hos Telenor.

---

# RAG: Retrieval Augmented Generation

- Vi hjelper en LLM til å komme med "riktig" svar ved å legge "relevante" deler av dokumenter (chunks) inn i konteksten, før den begynner å generere et svar.
- For å finne "relevant" del av dokumentet i dag, brukes gjerne embedding-modeller: Vi måler semantisk avstand mellom embeddingsvektoren for spørsmålet og embeddingsvektoren for små deler (chunks) av dokumenter.
- Embedding-modeller er også brukt i språkmodeller: Ordene vi skriver til Chat GPT blir gjort om til vektorer, før "hjernen" til Chat GPT behandler setningene videre.

---

# Embeddingsmodeller

- Input: Vi stapper en frase eller setning inn i modellen.
- Output: En vektor (array med flyttall), som representerer frasens "semantiske mening".
- (Egentlig: En vektor per ord/token, som blir normalisert.)
- To setninger som betyr omtrent det samme, får ca samme vektor.

**Vi skal se grundig og akademisk på hvordan dette fungerer, _etter_ en liten praktisk øvelse.**

---

# Måling av semantisk likhet: Kosinus-likhet

- Kosinus-likhet måler vinkelen mellom to vektorer (ikke avstanden!)
- Verdier fra -1 (motsatt) til 1 (identisk), 0 = ortogonal
- Formel: cos(θ) = (A · B) / (||A|| × ||B||)

**Hvorfor ikke Euklidisk avstand?**

- Kosinus ignorerer lengden på vektorer, fokuserer på retning
- "Katt er et dyr" og "Katter er dyr" bør være like, selv om lengden er forskjellig

```TypeScript
import { cos_sim } from '@huggingface/transformers'
```

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

- `git checkout https://github.com/lebbe/RMSB.git`

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

## Hvordan kan en vektor representere meningen i et ord eller en setning?

<table>
  <tr>
    <td></td>
    <td>Aardvark</td>
    <td>...</td>
    <td>computer</td>
    <td>data</td>
    <td>result</td>
    <td>pie</td>
    <td>sugar</td>
    <td>...</td>
  </tr>
  <tr>
    <td>cherry</td>
    <td>0</td>
    <td>...</td>
    <td>2</td>
    <td>8</td>
    <td>9</td>
    <td class="red">442</td>
    <td>25</td>
    <td>...</td>
  </tr>
  <tr>
    <td>strawberry</td>
    <td>0</td>
    <td>...</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td>60</td>
    <td>19</td>
    <td>...</td>
  </tr>
  <tr>
    <td>digital</td>
    <td>0</td>
    <td>...</td>
    <td>1670</td>
    <td>1683</td>
    <td>85</td>
    <td>5</td>
    <td>4</td>
    <td>...</td>
  </tr>
  <tr>
    <td>information</td>
    <td>0</td>
    <td>...</td>
    <td>3325</td>
    <td>3982</td>
    <td>378</td>
    <td>5</td>
    <td>13</td>
    <td>...</td>
  </tr>
</table>

Frekvens av ord som opptrer sammen i dokumenter i Wikipedia. Disse kan brukes
for å modellere ord, MEN: Utrolig glissent! Veldig mange nuller.

---

# Ord og semantikk

## Hvordan kan en vektor (array med flyttall) representere meningen i et ord eller en setning?

<table>
  <tr>
    <td></td>
    <td>Aardvark</td>
    <td>...</td>
    <td>computer</td>
    <td>data</td>
    <td>result</td>
    <td>pie</td>
    <td>sugar</td>
    <td>...</td>
  </tr>
  <tr>
    <td>cherry</td>
    <td>0</td>
    <td>...</td>
    <td>2</td>
    <td>8</td>
    <td>9</td>
    <td style="background: #c35">442</td>
    <td style="background: #c35">25</td>
    <td>...</td>
  </tr>
  <tr>
    <td>strawberry</td>
    <td>0</td>
    <td>...</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td style="background: #c35">60</td>
    <td style="background: #c35">19</td>
    <td>...</td>
  </tr>
  <tr>
    <td>digital</td>
    <td>0</td>
    <td>...</td>
    <td style="background: #35c">1670</td>
    <td style="background: #35c">1683</td>
    <td style="background: #35c">85</td>
    <td>5</td>
    <td>4</td>
    <td>...</td>
  </tr>
  <tr>
    <td>information</td>
    <td>0</td>
    <td>...</td>
    <td style="background: #35c">3325</td>
    <td style="background: #35c">3982</td>
    <td style="background: #35c">378</td>
    <td>5</td>
    <td>13</td>
    <td>...</td>
  </tr>
</table>

Vi ser at ordene "digital" og "information" er nærme hverandre semantisk. Så også "cherry" og "strawberry".

---

# Ord og semantikk

## Hvordan kan en vektor (array med flyttall) representere meningen i et ord eller en setning?

1. I gamle dager brukte vi altså glissne vektorer (mange nuller!) for å modellere semantikken til et ord.
2. I stedet kan vi benytte oss av embedding: Kort vektor for å representere et ord.
3. For eksempel word2vec!

---

# word2vec

I stedet for en gigantisk vektor, med like mange celler som det er ord i språket, lager vi en vilkårlig kort vektor (si 200) fylt med tilfeldige tall per ord.

word2vec bruker et nevralt nettverk som lærer å predikere kontekst-ord fra target-ord (eller omvendt), og vektorene er en bieffekt av denne læringen.

Dette er en eldre metode - dagens modeller (som vi bruker i oppgavene) er mer sofistikerte.

---

# Semantisk likhet mellom ord

Hvordan kan det at ord opptrer nærme hverandre kan bli semantisk like? La oss se på "katt" og "hund".

- Gutten klappet katten.
- Gutten klappet hunden.
- Hunden løp etter ballen.
- Katten løp etter musen.
- ....

Over tid, i et stort nok korpus, vil "hund" og "katt" være mer lik hverandre, enn ordene de opptrer sammen med.

---

# Mer om vektor-semantikk

- Moderne teknologier (embedding-modeller) bruker "nevrale nettverk" for å lage vektorer.
- De fungerer best om de opererer over fraser: Bruker kontekst for å skille mellom:
  - Jeg skal gi deg _bank_.
  - Jeg skal gå til en _bank_.
- De spytter ut en vektor per _token_.

---

<object data="ord.html" type="text/html" width="100%" height="100%">
  Åpne <a href="ord.html">ord.html</a> i eget vindu.
</object>

---

# Aritmetikk på semantiske vektorer (1)

- Vi kan måle semantikken i _fraser_ ved å legge sammen vektorene for ord!
- Da må vi huske å normalisere: Avstand fra origo til hvert punkt (token) er like langt.

<object data="normalisering.html" type="text/html" width="100%" height="60%">
  Åpne <a href="normalisering.html">normalisering.html</a> i eget vindu.
</object>

---

# Aritmetikk på semantiske vektorer (2)

Gitt vektorerene for ordene **a**, **male** og **person**:

$$\text{vektor for frasen} = \frac{\mathbf{a} + \mathbf{b} + \mathbf{c}}{n}$$

```
[0.12, 0.34, -0.23, 0.15, ...] + [0.40, 0.20, -0.34, 0.08, ...] + [0.18, 0.25, -0.19, 0.12, ...]

[0.12 + 0.40 + 0.18, 0.34 + 0.20 + 0.25, -0.23 - 0.34 - 0.19, ...]

=

[0.70, 0.79, -0.76, 0.35, ...]  / 3 =

[0.23, 0.26, -0.25, 0.12, ...]
```

Den normaliserte sammensatte vektoren representerer den semantiske betydningen av alle tre ordene sammen!

---

# Aritmetikk på semantiske vektorer (3)

Har vi vektoren for "konge", "mann" og "kvinne", så kan vi regne oss frem til vektoren for "dronning".

```
  king:   'the male monarch of a kingdom',
  male:   'a male person',
  female: 'a female person',
  queen:  'the female monarch of a kingdom'
```

```
Likhet mellom "king" og "male":          0.84
Likhet mellom "king" og "female":        0.76
Likhet mellom "male" og "female":        0.89
Likhet mellom "king" og "queen":         0.92

Likhet mellom "king - b + c" og "queen": 0.97
```

---

<object data="syntetisk_dronning.html" type="text/html" width="100%" height="100%">
  Åpne <a href="syntetisk_dronning.html">syntetisk_dronning.html</a> i eget vindu.
</object>

---

# RAG: Pipeline

0. Skaff et korpus vi kan bruke som kilde.
1. Del denne opp i chunks.
2. Lag en semantisk vektor for hver chunk.
3. Finn chunks som er semantisk mest lik et spørsmål (topp 3-5).
4. Få en LLM til å svare på spørsmålet med utgangspunkt i chunks.

---

# Oppgave 01: Chunking

1. Finn et større dokument du vil bruke som kilde.
2. Lagre denne som en .txt fil i `corpus`
3. Lag chunks av denne ved å fullføre koden i `01_chunking.ts`

**Tips:** Typisk chunk-størrelse er 256-512 tokens med ~10% overlap for bedre resultat.

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

1. Les inn alle embeddings lagd i Oppgave 2.
2. Lag en embedding for et spørsmål `question`.
3. Mål kosinus-likhet mellom spørsmålet og alle chunks med `cos_sim`.
4. `console.log` ut de chunks som "ligner" mest på spørsmålet, for eksempel topp 4.

---

# Andre morsomme ting å bruke embedding til

- Semantisk caching: I stedet for å generere det samme svaret på nytt igjen,
  for et "semantisk" likt spørsmål, hent opp svaret fra en semantisk cache.
- Finn duplicate support-tickets, bug-rapporter, etc.
- Anbefalingssystemer: Link til artikler med chunks som er "lik" chunks i denne artikkelen.
- Automatisk kategorisering av eposter, tickets, etc.

---

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

- **pinecone / Weaviate / Qdrant**: Vector-databaser
- **pgvector**: Postgres-utvidelse hvis du allerede bruker Postgres
- **Vertex AI Search / Azure AI Search**: Fullstendige managed løsninger
- **Hybrid search**: Kombiner semantisk + keyword (BM25) for bedre recall
- **Reranking**: Re-rank top-k med en annen modell for økt presisjon

# Kilder

- https://web.stanford.edu/~jurafsky/slp3/
