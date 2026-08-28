import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import TextLabApp from './TextLabApp'
import './text-lab.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TextLabApp />
  </StrictMode>,
)

