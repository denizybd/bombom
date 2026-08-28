import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        sinaps: 'index.html',
        metinLaboratuvari: 'metin-laboratuvari/index.html',
        noronSentezi: 'noron-sentezi/index.html',
        noronAileleri: 'noron-aileleri/index.html',
      },
    },
  },
  test: {
    environment: 'node',
    globals: true,
    coverage: {
      reporter: ['text', 'html'],
    },
  },
})
