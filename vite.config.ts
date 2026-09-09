import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    rollupOptions: {
      input: {
        sinaps: 'index.html',
        metinLaboratuvari: 'metin-laboratuvari/index.html',
        noronSentezi: 'noron-sentezi/index.html',
        noronAileleri: 'noron-aileleri/index.html',
        otonomEgitim: 'otonom-egitim/index.html',
        dusunceAkisi: 'dusunce-akisi/index.html',
        kavramFidanligi: 'kavram-fidanligi/index.html',
        bolgeAtlasi: 'bolge-atlasi/index.html',
        noronEvreni: 'noron-evreni/index.html',
      },
    },
  },
  test: {
    environment: 'node',
    globals: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    coverage: {
      reporter: ['text', 'html'],
    },
  },
})
