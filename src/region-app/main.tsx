import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RegionApp } from './RegionApp'

createRoot(document.getElementById('root')!).render(<StrictMode><RegionApp /></StrictMode>)
