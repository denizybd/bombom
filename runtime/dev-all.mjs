import { spawn } from 'node:child_process'
import path from 'node:path'
import { build, createServer } from 'vite'

const root = path.resolve(import.meta.dirname, '..')
const uiUrl = 'http://127.0.0.1:5173/otonom-egitim/'

async function trainingServiceAlreadyRunning() {
  try {
    const response = await fetch('http://127.0.0.1:8787/api/health', { signal: AbortSignal.timeout(750) })
    return response.ok
  } catch {
    return false
  }
}

async function uiAlreadyRunning() {
  try {
    const response = await fetch(uiUrl, { signal: AbortSignal.timeout(750) })
    return response.ok
  } catch {
    return false
  }
}

function openBrowserWhenRequested() {
  if (process.env.BOMBOM_OPEN_BROWSER !== '1' || process.platform !== 'darwin') return
  const opener = spawn('open', [uiUrl], { detached: true, stdio: 'ignore' })
  opener.unref()
}

const serviceWasRunning = await trainingServiceAlreadyRunning()
const uiWasRunning = await uiAlreadyRunning()

if (serviceWasRunning && uiWasRunning) {
  console.log(`Bombom zaten çalışıyor: ${uiUrl}`)
  openBrowserWhenRequested()
  process.exit(0)
}

let trainingProcess = null
if (!serviceWasRunning) {
  await build({ configFile: path.join(root, 'vite.training.config.ts') })
  trainingProcess = spawn(process.execPath, [path.join(root, 'dist-training-service/training-service.mjs')], {
    cwd: root,
    stdio: 'inherit',
    env: process.env,
  })
}

let ui = null
if (!uiWasRunning) {
  ui = await createServer({ configFile: path.join(root, 'vite.config.ts'), root })
  await ui.listen()
  ui.printUrls()
} else {
  console.log(`Görsel uygulama zaten çalışıyor: ${uiUrl}`)
}
openBrowserWhenRequested()

let closing = false
async function shutdown() {
  if (closing) return
  closing = true
  await ui?.close()
  trainingProcess?.kill('SIGTERM')
}

process.once('SIGINT', () => void shutdown())
process.once('SIGTERM', () => void shutdown())
trainingProcess?.once('exit', (code) => {
  if (!closing && code) console.error(`Otonom eğitim servisi ${code} koduyla durdu.`)
})
