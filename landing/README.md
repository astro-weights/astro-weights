# Landing page

Source of https://astro-weights.dev/ (docs live under `/docs/`). `src/` and `shared/` are verbatim copies from the
`astro-weights-site` repo (mock `04-three-burns`); do not edit them here. Re-copy:

```sh
rsync -a --delete --exclude NOTES.md --exclude index.html ../astro-weights-site/mocks/04-three-burns/ landing/src/
cp ../astro-weights-site/shared/{logo.svg,tokens.css} landing/shared/
```

`index.html`, `vite.config.js` and `package.json` are owned by this repo. Build: `scripts/build-site.sh`.
