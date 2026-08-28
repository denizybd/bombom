import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SynthesisApp } from './SynthesisApp'
import './synthesis-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SynthesisApp />
  </StrictMode>,
)
