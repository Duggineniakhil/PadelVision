# PadelVision web app

Next.js (App Router) + TypeScript + Tailwind. It talks to the local API in `../api`
(`NEXT_PUBLIC_API_URL`, default `http://127.0.0.1:8000`).

```bash
npm install
npm run dev          # http://localhost:3000 (start the API first: padelvision-api)
npm run lint
npx tsc --noEmit
npm run gen:api      # regenerate src/lib/api-types.ts after: python ../scripts/export_openapi.py
```

Structure:

```
src/app/                    routes: / (matches), /calibrate, /matches/[id]/{,video,players,rallies,shots,highlights,court}
src/components/ui.tsx       buttons, cards, stats, callouts, dialog
src/components/shell.tsx    header, import dialog, command palette (Ctrl+K)
src/components/court.tsx    court diagrams (SVG, metres): heatmaps, shot maps, landing points
src/components/timeline.tsx match timeline of rallies
src/components/calibration.tsx  court calibration editor (click + snap court lines, live fit and accuracy)
src/lib/api.ts              typed API client + SWR hooks (types generated from the API's OpenAPI schema)
```

Video overlays are drawn on a canvas over the original video from the run's JSON, frame-synchronised with
`requestVideoFrameCallback`.
