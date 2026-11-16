// Header site name links to the landing page, like the logo (extra.homepage in zensical.toml).
document.querySelectorAll('.md-header__title .md-header__topic:first-child .md-ellipsis').forEach((el) => {
  const a = document.createElement('a')
  a.href = 'https://astro-weights.dev/'
  a.textContent = el.textContent.trim()
  a.style.color = 'inherit'
  el.replaceChildren(a)
})
