import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react'

const CONFIG = {
  Minor:    { cls: 'badge-minor',    Icon: ShieldCheck,   label: 'Minor Interaction' },
  Moderate: { cls: 'badge-moderate', Icon: AlertTriangle, label: 'Moderate Interaction' },
  Major:    { cls: 'badge-major',    Icon: AlertOctagon,  label: 'Major Interaction' },
}

export default function SeverityBadge({ level, size = 'md', showLabel = false }) {
  const cfg = CONFIG[level] || CONFIG.Moderate
  const { cls, Icon, label } = cfg
  const iconSize = size === 'lg' ? 16 : size === 'sm' ? 12 : 13

  const style = size === 'lg'
    ? { padding: '0.4375rem 1rem', fontSize: '0.875rem', borderRadius: '6px' }
    : size === 'sm'
    ? { padding: '0.125rem 0.5rem', fontSize: '0.6875rem' }
    : {}

  return (
    <span className={`badge ${cls}`} style={style} role="status" aria-label={`Severity: ${level}`}>
      <Icon size={iconSize} aria-hidden="true" />
      {showLabel ? label : level}
    </span>
  )
}

export function SeverityDot({ level }) {
  const colors = {
    Minor:    'var(--color-minor)',
    Moderate: 'var(--color-moderate)',
    Major:    'var(--color-major)',
  }
  return (
    <span
      style={{
        display: 'inline-block',
        width: 8, height: 8,
        borderRadius: '50%',
        background: colors[level] || colors.Moderate,
        flexShrink: 0,
      }}
      aria-hidden="true"
    />
  )
}
