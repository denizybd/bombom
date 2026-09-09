// @ts-nocheck -- This Vite-only configuration runs in Node; the app intentionally omits Node typings.
import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

const require = createRequire(import.meta.url)
const dictionaryDirectory = path.dirname(require.resolve('dictionary-tr'))

const copyTurkishDictionary = {
  name: 'copy-turkish-hunspell-dictionary',
  generateBundle(this: { emitFile: (asset: { type: 'asset'; fileName: string; source: Buffer }) => void }) {
    for (const fileName of ['index.aff', 'index.dic']) {
      const source = readFileSync(path.join(dictionaryDirectory, fileName))
      this.emitFile({ type: 'asset', fileName, source })
    }
  },
}

export default defineConfig({
  plugins: [copyTurkishDictionary],
  ssr: {
    noExternal: ['nspell', 'dictionary-tr'],
  },
  build: {
    ssr: true,
    outDir: 'dist-training-service',
    emptyOutDir: true,
    target: 'node24',
    minify: false,
    sourcemap: true,
    rollupOptions: {
      input: {
        'training-service': 'runtime/start.mjs',
        'training-worker': 'runtime/training-worker.mjs',
        'pipeline-worker': 'runtime/pipeline-worker.mjs',
      },
      output: {
        entryFileNames: '[name].mjs',
        // dictionary-tr resolves its files relative to import.meta.url. Keep
        // shared runtime chunks beside the single dictionary copy rather than
        // duplicating the 9 MiB dictionary under assets/ for each location.
        chunkFileNames: '[name]-[hash].mjs',
      },
    },
  },
})
