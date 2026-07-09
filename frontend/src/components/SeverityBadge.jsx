import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react'

const CONFIG = {
  Minor:    { cls: 'badge-minor',    Icon: ShieldCheck,    color: 'var(--color-minor)'    },
  Moderate: { cls: 'badge-moderate', Icon: AlertTriangle,  color: 'var(--color-moderate)' },
  Major:    { cls: 'badge-major',    Icon: AlertOctagon,   color: 'var(--color-major)'    },
}

export default function SeverityBadge({ level, size = 'md' }) {
  const cfg = CONFIG[level] || CONFIG.Moderate
  const { cls, Icon } = cfg
  const iconSize = size === 'lg' ? 18 : 14
  const style = size === 'lg' ? { padding: '0.5rem 1.25rem', fontSize: '0.95rem' } : {}

  return (
    <span className={`badge ${cls}`} style={style}>
      <Icon size={iconSize} />
      {level}
    </span>
  )
}

export function SeverityDot({ level }) {
  const colors = { Minor: 'var(--color-minor)', Moderate: 'var(--color-moderate)', Major: 'var(--color-major)' }
  return (
    <span style={{
      display: 'inline-block',
      width: 8, height: 8,
      borderRadius: '50%',
      background: colors[level] || colors.Moderate,
      boxShadow: `0 0 6px ${colors[level] || colors.Moderate}`,
    }} />
  )
}
