import {
  BrainCircuit,
  Bot,
  Check,
  ChevronDown,
  FlaskConical,
  Orbit,
  Route,
  ScanSearch,
  Sparkles,
  UsersRound,
  Waves,
  type LucideIcon,
} from 'lucide-react'
import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react'
import './suite-menu.css'

export type SuiteAppId = 'sinaps' | 'text-lab' | 'synthesis' | 'families' | 'training' | 'thought' | 'concepts' | 'regions' | 'universe'

interface SuiteApp {
  id: SuiteAppId
  label: string
  description: string
  href: string
  icon: LucideIcon
}

const SUITE_APPS: readonly SuiteApp[] = [
  {
    id: 'sinaps',
    label: 'Sinaps',
    description: 'Olasılık ağı ve sinyal simülasyonu',
    href: '/',
    icon: BrainCircuit,
  },
  {
    id: 'text-lab',
    label: 'Metin Laboratuvarı',
    description: 'Ham metinden değişmez temsil üretimi',
    href: '/metin-laboratuvari/',
    icon: FlaskConical,
  },
  {
    id: 'synthesis',
    label: 'Nöron Sentezi',
    description: 'Yüzey kalıbı doğum adayları',
    href: '/noron-sentezi/',
    icon: Sparkles,
  },
  {
    id: 'families',
    label: 'Nöron Aileleri',
    description: 'Davranış, aidiyet ve koalisyon',
    href: '/noron-aileleri/',
    icon: UsersRound,
  },
  {
    id: 'training',
    label: 'Otonom Eğitim',
    description: 'Ollama öğretmeni ve gelişim döngüsü',
    href: '/otonom-egitim/',
    icon: Bot,
  },
  {
    id: 'thought',
    label: 'Düşünce Akışı',
    description: 'Sinyalin tik tik dallanan yolculuğu',
    href: '/dusunce-akisi/',
    icon: Waves,
  },
  {
    id: 'concepts',
    label: 'Kavram Fidanlığı',
    description: 'Tekrarlanan yollar, makrolar ve shadow kanıtı',
    href: '/kavram-fidanligi/',
    icon: Route,
  },
  {
    id: 'regions',
    label: 'Plastik Bölge Atlası',
    description: 'Nöron içi mikro-sayfa aidiyeti ve yaşam döngüsü',
    href: '/bolge-atlasi/',
    icon: ScanSearch,
  },
  {
    id: 'universe',
    label: 'Nöron Evreni 3B',
    description: 'Nöron, aile ve akışların salt okunur haritası',
    href: '/noron-evreni/',
    icon: Orbit,
  },
]

interface SuiteMenuProps {
  currentApp: SuiteAppId
  children: ReactNode
  navigationLocked?: boolean
}

export function SuiteMenu({ currentApp, children, navigationLocked = false }: SuiteMenuProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const menuId = useId()

  const focusFirstItem = () => {
    window.requestAnimationFrame(() => {
      menuRef.current?.querySelector<HTMLElement>('[role="menuitem"]:not([aria-disabled="true"])')?.focus()
    })
  }

  useEffect(() => {
    if (!open) return

    const closeFromOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const closeFromEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setOpen(false)
      triggerRef.current?.focus()
    }

    document.addEventListener('pointerdown', closeFromOutside)
    document.addEventListener('keydown', closeFromEscape)
    return () => {
      document.removeEventListener('pointerdown', closeFromOutside)
      document.removeEventListener('keydown', closeFromEscape)
    }
  }, [open])

  const handleTriggerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== 'ArrowDown') return
    event.preventDefault()
    setOpen(true)
    focusFirstItem()
  }

  const handleMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    event.preventDefault()
    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([aria-disabled="true"])') ?? [],
    )
    if (!items.length) return
    const currentIndex = items.indexOf(document.activeElement as HTMLElement)
    const step = event.key === 'ArrowDown' ? 1 : -1
    const nextIndex = currentIndex < 0
      ? (step > 0 ? 0 : items.length - 1)
      : (currentIndex + step + items.length) % items.length
    items[nextIndex]?.focus()
  }

  return (
    <div className={`suite-menu ${open ? 'is-open' : ''}`} ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="suite-menu__trigger"
        aria-label="Laboratuvar menüsünü aç"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={menuId}
        title="Laboratuvarlar"
        onClick={() => setOpen((current) => !current)}
        onKeyDown={handleTriggerKeyDown}
      >
        {children}
        <ChevronDown className="suite-menu__chevron" size={15} aria-hidden="true" />
      </button>

      {open && (
        <div
          ref={menuRef}
          id={menuId}
          className="suite-menu__popover"
          role="menu"
          aria-label="Laboratuvarlar"
          onKeyDown={handleMenuKeyDown}
        >
          <div className="suite-menu__heading">
            <span>ÇALIŞMA ALANLARI</span>
            <small>{navigationLocked ? 'İşlem sürerken geçiş kilitli' : 'Bir laboratuvar seç'}</small>
          </div>
          <div className="suite-menu__list">
            {SUITE_APPS.map((app) => {
              const Icon = app.icon
              const current = app.id === currentApp
              const body = (
                <>
                  <span className="suite-menu__item-icon" data-app={app.id}><Icon size={18} /></span>
                  <span className="suite-menu__item-copy"><b>{app.label}</b><small>{app.description}</small></span>
                  {current && <Check className="suite-menu__check" size={16} aria-label="Şu an açık" />}
                </>
              )

              if (current) {
                return <button key={app.id} type="button" role="menuitem" className="suite-menu__item is-current" onClick={() => setOpen(false)}>{body}</button>
              }
              if (navigationLocked) {
                return <span key={app.id} role="menuitem" aria-disabled="true" className="suite-menu__item is-disabled">{body}</span>
              }
              return <a key={app.id} role="menuitem" className="suite-menu__item" href={app.href} onClick={() => setOpen(false)}>{body}</a>
            })}
          </div>
          <div className="suite-menu__footer"><span className={navigationLocked ? 'is-busy' : ''} /><small>{navigationLocked ? 'Etkin işlem korunuyor' : 'Yerel çalışma alanı'}</small></div>
        </div>
      )}
    </div>
  )
}
