export interface ProfileBarItem {
  id: string
  label: string
  value: number
  group: string
  comparison?: number
}

export function ProfileBars({
  items,
  label,
}: {
  items: readonly ProfileBarItem[]
  label: string
}) {
  const groups = Array.from(new Set(items.map((item) => item.group)))
  return (
    <div className="fa-profile-bars" aria-label={label}>
      {groups.map((group) => (
        <section key={group}>
          <h3>{group}</h3>
          {items.filter((item) => item.group === group).map((item) => {
            const value = Math.min(1, Math.max(0, item.value))
            const comparison = item.comparison === undefined ? null : Math.min(1, Math.max(0, item.comparison))
            return (
              <div className="fa-profile-bar" key={item.id}>
                <div><span>{item.label}</span><b>{item.value.toFixed(4)}</b></div>
                <div className="fa-profile-bar__track" role="meter" aria-label={item.label} aria-valuemin={0} aria-valuemax={1} aria-valuenow={value}>
                  <i style={{ width: `${value * 100}%` }} />
                  {comparison !== null && <em style={{ left: `${comparison * 100}%` }} title={`Karşılaştırma ${comparison.toFixed(4)}`} />}
                </div>
              </div>
            )
          })}
        </section>
      ))}
    </div>
  )
}
