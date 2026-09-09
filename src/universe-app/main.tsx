import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { UniverseApp } from './UniverseApp'
import './universe-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <UniverseApp />
  </StrictMode>,
)
