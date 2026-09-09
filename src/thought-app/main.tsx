import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThoughtApp } from './ThoughtApp'
import './thought-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThoughtApp />
  </StrictMode>,
)
