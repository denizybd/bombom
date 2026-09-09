import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { TrainingApp } from './TrainingApp'
import './training-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TrainingApp />
  </StrictMode>,
)
