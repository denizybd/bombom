import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { FamilyApp } from './FamilyApp'
import './family-app.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <FamilyApp />
  </StrictMode>,
)
