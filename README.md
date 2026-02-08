# RAG med strikk og binders i TypeScript

Velkommen til prosjektet der vi binder sammen LLM, embeddings og semantikk med strikk, binders og en dæsj TypeScript.

## Hva er greia?

Her bygger vi en RAG-pipeline fra bunnen av:

- Vi deler opp et dokument i små biter (chunks).
- Vi lager vektorer som fanger meningen i hver bit.
- Vi måler likhet mellom fraser, sånn at maskinen kan skjønne hva som ligner.
- Vi lar en LLM svare på spørsmål, med utgangspunkt i relevante tekstbiter.

## Kom i gang

1. Installer avhengigheter:

   ```bash
   npm i
   ```

2. Lek med eksempelkoden:

   ```bash
   npx tsx src/00_eksempel.ts
   ```

   Sjekk semantisk likhet mellom fraser du måtte lure på. Modellen kan ta litt tid å laste ned, så ta deg gjerne en kopp kaffe.

## Pipeline

0. Skaff et korpus vi kan bruke som kilde.
1. Del denne opp i chunks.
2. Lag en semantisk vektor for hver chunk.
3. Finn chunks som er semantisk mest lik et spørsmål.
4. Få en LLM til å svare på spørsmålet med utgangspunkt i chunks.

## Oppgaver

- Se på filene i `src/`
- Fullfør kodestubbene, og la maskinen gjøre jobben.
- Spør deg selv: Hva er egentlig "semantikk"?

## Going prod

Når du har fått alt til å funke, kan du utforske:

- @pinecone-database/pinecone: Vektordatabase for prod.
- Vertex AI Search: Fullstendig RAG-pipeline i Google Cloud.

---

Lykke til!

_Lars-Erik Bruce / Redpill Linpro_
