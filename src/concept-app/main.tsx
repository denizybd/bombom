import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ConceptApp } from './ConceptApp'
import './concept-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConceptApp />
  </StrictMode>,
)
