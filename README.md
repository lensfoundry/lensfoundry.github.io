# lensprep.github.io

Source of the lensprep lens-image browser: one static page over the HST and JWST strong-lens
reductions ([HST_lenses_data_reduction](https://github.com/samlange04/HST_lenses_data_reduction),
[JWST_lenses_data_reduction](https://github.com/samlange04/JWST_lenses_data_reduction)).

```bash
uv run --project ../HST_lenses_data_reduction python build.py --out _site
```

`build.py` reads only the files tracked in git in the two reduction checkouts beside this
folder, and writes `_site/index.html` (`template.html` with the catalogue embedded) plus one
four-panel sprite per lens and band in `_site/img/` (image, lensed light with an
elliptical-isophote model of the deflector removed, noise map, PSF; North up, East left).
HST lenses are grouped as in the reduction repo: six samples, each split into single- or
multi-band lenses and the categories prime, complex, limited and multi-galaxy, tagged on every
card (e.g. `MB-P`). File links point at GitHub until
the Zenodo records exist. Prototype: not yet published to GitHub Pages.
