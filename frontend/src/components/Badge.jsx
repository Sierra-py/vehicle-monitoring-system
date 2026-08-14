const VARIANT_MAP = {
  approved: 'ok',
  ok: 'ok',
  exact: 'ok',
  fuzzy: 'ok',
  pending: 'pending',
  rejected: 'bad',
  bad: 'bad',
  ambiguous: 'pending',
  no_match: 'neutral',
  inside: 'ok',
  outside: 'neutral',
}

export function Badge({ children, variant }) {
  const resolved = variant || VARIANT_MAP[String(children).toLowerCase()] || 'neutral'
  return <span className={`badge ${resolved}`}>{children}</span>
}
