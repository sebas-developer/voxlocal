# @voxlocal/client

TypeScript client for the optional VoxLocal HTTP API.

```bash
npm install
npm run build
```

```ts
import { stream, transcribe } from "@voxlocal/client";

const transcript = await transcribe(
  "http://127.0.0.1:8765",
  await wavFile.arrayBuffer(),
);
console.log(transcript);

for await (const chunk of stream("http://127.0.0.1:8765", {
  language: "es",
  text: "Hola mundo",
})) {
  console.log(chunk.sequence, chunk.sample_rate, chunk.final);
}
```
