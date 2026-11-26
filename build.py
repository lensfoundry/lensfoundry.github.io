#!/usr/bin/env python
"""Build the lensfoundry lens-image browser: one static page over the HST and JWST reductions.

    uv run --project ../HST_lenses_data_reduction python build.py [--out _site] [--only LENS]

Reads ONLY files tracked in git in the two reduction repos (the stamps, masks, PSFs and
positions under data/stamps/, and the info/ JSONs), so the browser shows exactly what a clone
carries — with ONE exception: the PSF panel renders the full kernel (with wings) from the
gitignored data/psf/ tree when it exists on disk, falling back to the trimmed stamp kernel
otherwise. Writes

    <out>/index.html            the page: template.html with the catalogue embedded as JSON
    <out>/img/<lens>__<band>.jpg one sprite per band: image | lensed light | masked S/N | noise | PSF

The HST stamps sit at data/stamps/<sample>/<band_mode>/<category>/<lens>/<band>/ (since
2026-11-02), the JWST ones still at data/stamps/<sample>/<lens>/<band>/; the lens and band are
always the last two path parts.

Panels are drawn North up, East left (FITS column to the right, row upwards), so a (col, row)
quoted in a review note counts from the bottom-left corner. The "lensed light" panel subtracts
an elliptical-isophote model of the deflector (the HST repo's scripts/isophote_model.py, the
same view as its positions GUI): a finding aid, never photometry.

Download links for HST point at the Zenodo record (concept DOI 10.5281/zenodo.24052764),
using the version-pinned URLs in the HST repo's info/zenodo_archives.csv (the step-5
fetch manifest); JWST keeps GitHub raw links until its own record exists.
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import subprocess
from urllib.parse import quote

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
import sys
sys.path.insert(0, os.path.join(CODE, 'HST_lenses_data_reduction', 'scripts'))
import isophote_model  # noqa: E402
PANEL = 480          # px per panel side in the sprite
THUMB = 220          # px, lens-card thumbnail (central crop)

TELESCOPES = [
    dict(key='hst', name='Hubble Space Telescope', short='HST',
         path=os.path.join(CODE, 'HST_lenses_data_reduction'),
         gh='samlange04/HST_lenses_data_reduction'),
    dict(key='jwst', name='James Webb Space Telescope', short='JWST',
         path=os.path.join(CODE, 'JWST_lenses_data_reduction'),
         gh='samlange04/JWST_lenses_data_reduction'),
]

SAMPLES = {
    'slacs': ('SLACS', 'Lenses from the Sloan Lens ACS Survey: early-type deflectors at z 0.06–0.5 with emission-line sources.',
              'Bolton et al. 2008; Auger et al. 2009; Etherington et al. 2022'),
    'swells': ('SWELLS', 'Disk-galaxy deflectors from SWELLS that SLACS also found.',
               'Treu et al. 2011; Brewer et al. 2012'),
    'gallery': ('BELLS GALLERY', 'Lensed Lyman-alpha emitters at z 2–3 behind BOSS galaxies, imaged with WFC3/UVIS.',
                'Shu et al. 2016'),
    'bells': ('BELLS', 'BOSS emission-line lenses imaged with ACS in F814W.',
              'Brownstein et al. 2012'),
    'dinos': ('SL2S + Project Dinos', 'SL2S lenses re-imaged in deep WFC3/UVIS F475X by Project Dinos.',
              'Sonnenfeld et al. 2013; Sheu et al. 2025'),
    'sl2s': ('SL2S', 'The other SL2S lenses: older WFPC2 F606W or short F475X + F600LP snapshots.',
             'Sonnenfeld et al. 2013'),
    'b1938': ('JVAS B1938+666', 'A radio-loud Einstein ring with 21 dithers per filter, ~24 h each.',
              'GO 8281 (PI Vegetti)'),
    'cowls_spectacular': ('COWLS spectacular', 'The 17 top-tier lenses of the COSMOS-Web Lens Survey, imaged with NIRCam.',
                          'Mahler et al. 2025 (COWLS II)'),
    'spt': ('SPT dusty lenses', 'Two lensed dusty star-forming galaxies from the TEMPLATES Early Release Science program.',
            'Rigby et al. 2025 (TEMPLATES)'),
}

# Published Einstein radii for the JWST lenses (the JWST repo keeps none in info/; AGENTS.md
# quotes these).
JWST_THETA_E = {
    'B1938+666': (0.45, 'radio ring, approximate'),
    'SPT0418-47': (1.207, 'Cathey et al. 2024'),
    'SPT2147-50': (1.195, 'Spilker et al. 2016'),
}

# Pivot wavelengths in microns: band chips are coloured by wavelength.
PIVOT = {'F438W': 0.433, 'F475X': 0.494, 'F555W': 0.536, 'F606W': 0.589, 'F600LP': 0.746,
         'F814W': 0.80, 'F115W': 1.154, 'F150W': 1.501, 'F160W': 1.537, 'F200W': 1.99,
         'F277W': 2.79, 'F356W': 3.57, 'F444W': 4.42}
# Which band a lens card shows: reddest optical for HST, LW for JWST.
CARD_BAND = ['F814W', 'F606W', 'F555W', 'F600LP', 'F475X', 'F160W', 'F438W',
             'F444W', 'F277W', 'F356W', 'F200W', 'F150W', 'F115W']

# Preferred sample display order on the home page (slacs and gallery first, Sam 2026-11-26).
SAMPLE_ORDER = ['slacs', 'gallery']

INSTRUMENT = {'ACS/WFC': 'ACS/WFC', 'WFPC2/WF3': 'WFPC2 (WF3 chip)', 'WFPC2/PC': 'WFPC2 (WF3 chip)',
              'WFC3/IR': 'WFC3/IR', 'WFC3/UVIS': 'WFC3/UVIS',
              'NIRCAM/SW': 'NIRCam short wavelength', 'NIRCAM/LW': 'NIRCam long wavelength'}


# ─────────────────────────────── helpers ────────────────────────────────

def tracked(repo):
    out = subprocess.run(['git', '-C', repo, 'ls-files', 'data/stamps', 'info'],
                         capture_output=True, text=True, check=True).stdout
    return set(out.split('\n')) - {''}


def load(repo, rel):
    path = os.path.join(repo, 'info', rel)
    return json.load(open(path)) if os.path.isfile(path) else {}


def load_zenodo_archives(repo):
    """archive name -> {url, size_bytes} from the HST repo's step-5 fetch manifest
    (info/zenodo_archives.csv; version-pinned Zenodo URLs). Empty when absent."""
    path = os.path.join(repo, 'info', 'zenodo_archives.csv')
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {r['archive']: r for r in csv.DictReader(f)}


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def band_archive(zarch, sample, band):
    """The smallest Zenodo archive holding this band, as a display dict, or None for a
    telescope with no record yet: the per-(sample, filter) zip, else the sample zip
    (bells, one filter, has only bells_all.zip)."""
    for name in (f"{sample}_{band.split('_')[0]}.zip", f'{sample}_all.zip'):
        if name in zarch:
            row = zarch[name]
            return dict(name=name, url=row['url'], size=human(int(row['size_bytes'])))
    return None


def gh_link(tel, path):
    """A browse-the-file-on-GitHub link for one data file (genuine per-file view; the
    big archive download is presented separately, not behind a file name)."""
    return dict(url=f"https://github.com/{tel['gh']}/raw/main/{quote(path)}", where='GitHub ↗')


def token(name):
    return name.replace('+', 'p')


def band_label(band):
    m = re.match(r'([fF]\d+[A-Za-z]+)(?:_v(\d+))?$', band)
    base = m.group(1).upper() if m else band.upper()
    return base + (f' (visit {m.group(2)})' if m and m.group(2) else '')


def base_band(band):
    return band.split('_')[0].upper()


def short_cite(source):
    """'Auger+2009' from a catalogue row's free-text source, for a compact citation."""
    m = re.search(r"[A-Z][A-Za-zéë'()-]+\+\d{4}", source or '')
    return m.group(0) if m else None


# ADS abstract pages for the papers the catalogue cites (short_cite form -> bibcode).
_CITE_BIBCODES = {
    'Auger+2009': '2009ApJ...705.1099A',       # SLACS IX
    'Brownstein+2012': '2012ApJ...744...41B',  # BELLS I
    'Shu+2016': '2016ApJ...833..264S',         # BELLS GALLERY IV
    'Sonnenfeld+2013': '2013ApJ...777...97S',  # SL2S III
    'Sheu+2025': '2025MNRAS.541....1S',        # Project Dinos II
    'Brewer+2012': '2012MNRAS.422.3574B',      # SWELLS III
    'Treu+2011': '2011MNRAS.417.1601T',        # SWELLS I
    'Gavazzi+2014': '2014ApJ...785..144G',
    'DESI DR1': '2024AJ....168...58D',
}


def cite_obj(row, key=None):
    """{'text', 'url'} for a catalogue value's citation: the row's own per-field source
    (`key`, a dict like zlens_src or a free-text string) where set, else the row-level
    source short-formed. url is the paper's ADS page where known."""
    v = row.get(key) if key else None
    if isinstance(v, dict):
        text = v.get('cite') or short_cite(v.get('source'))
    elif isinstance(v, str):
        text = v
    else:
        text = None
    text = text or short_cite(row.get('source'))
    if not text:
        return None
    bib = _CITE_BIBCODES.get(text)
    return dict(text=text, url=f'https://ui.adsabs.harvard.edu/abs/{bib}/abstract' if bib else None)


def band_colour(band):
    lam = PIVOT.get(base_band(band))
    if lam is None:
        return '#888888'
    t = np.clip((np.log10(lam) - np.log10(0.42)) / (np.log10(4.5) - np.log10(0.42)), 0, 1)
    # blue -> teal -> yellow -> orange -> deep red, in log wavelength
    stops = [(0.0, (76, 123, 230)), (0.22, (52, 178, 170)), (0.38, (214, 196, 72)),
             (0.62, (232, 132, 56)), (1.0, (196, 58, 74))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            c = [round(a + f * (b - a)) for a, b in zip(c0, c1)]
            return '#%02x%02x%02x' % tuple(c)
    return '#%02x%02x%02x' % stops[-1][1]


def mad_sigma(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return 1.4826 * np.median(np.abs(x - np.median(x))) or (np.std(x) or 1.0)


def render(arr, cmap, vmin, vmax, overlay=None):
    """One square panel as an (PANEL, PANEL, 3) uint8 array. overlay(ax) draws on top."""
    fig = plt.figure(figsize=(PANEL / 100, PANEL / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_ylim(-0.5, arr.shape[0] - 0.5)
    ax.set_axis_off()
    if overlay:
        overlay(ax)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


HALO = [pe.Stroke(linewidth=3.2, foreground='black'), pe.Normal()]


SET_COLOURS = ['#6ff0ff', '#ff86d8', '#a6f36a', '#ffb347']   # positions: source 1, 2, ...


def panels(sci, noise, psf, mask, position_sets_px, theta_px, scale, centre=None, psf_full=None):
    n = sci.shape[0]
    finite = np.isfinite(sci) & np.isfinite(noise) & (noise < 1e7)
    sci = np.where(finite, sci, np.nan)
    mask = mask if mask is not None else np.zeros_like(sci, bool)
    yy, xx = np.indices(sci.shape)
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)

    outer = finite & ~mask & (r > 0.35 * n)
    sky = np.nanmedian(sci[outer]) if outer.any() else np.nanmedian(sci)
    sig = mad_sigma(sci[outer] if outer.any() else sci)

    # 1. image: asinh about the sky
    a = 2.0 * sig
    img = np.arcsinh((sci - sky) / a)
    core = finite & (r < 0.25 * n)
    hi = np.nanpercentile(img[core], 99.9) if core.any() else np.nanmax(img)
    p1 = render(np.nan_to_num(img, nan=-9), 'inferno', np.arcsinh(-1.5 * sig / a), hi)

    # 2. lensed light: an elliptical-isophote model of the deflector removed (the positions
    # GUI's own view), on the S/N image, lightly smoothed; red = light the model lacks.
    snr = np.where(finite, (sci - sky) / np.where(noise > 0, noise, np.inf), 0.0)
    res, vdiv = isophote_model.isophote_display(snr, scale, exclude=mask | ~finite, centre=centre)
    def over(ax):
        if mask.any():
            shade = np.zeros(mask.shape + (4,))
            shade[mask] = (0.0, 0.0, 0.0, 0.58)
            ax.imshow(shade, origin='lower', interpolation='nearest')
            cs = ax.contour(mask.astype(float), levels=[0.5], colors='white', linewidths=1.1)
            for c in getattr(cs, 'collections', [cs]):
                c.set_path_effects(HALO)
        if theta_px:
            cx, cy = centre if centre is not None else ((n - 1) / 2, (n - 1) / 2)
            circ = plt.Circle((cx, cy), theta_px, fill=False, color='black',
                              lw=1.1, ls=(0, (4, 3)), alpha=0.85)
            ax.add_patch(circ)
        for k, pts in enumerate(position_sets_px):
            for col, row in pts:
                ax.plot([col], [row], marker='+', ms=17, mew=2.0, color=SET_COLOURS[k % len(SET_COLOURS)],
                        path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
    p2 = render(np.nan_to_num(res, nan=0), 'RdBu_r', -vdiv, vdiv, over)

    # 3. masked S/N: what a fit sees -- the science image over the noise map, with the
    # contaminant mask applied.
    a_snr = 2.0
    lsnr = np.arcsinh(np.where(mask, np.nan, snr) / a_snr)
    finite_snr = lsnr[np.isfinite(lsnr)]
    snr_hi = np.nanpercentile(finite_snr, 99.5) if finite_snr.size else 1.0
    p3 = render(np.nan_to_num(lsnr, nan=-2), 'inferno', np.arcsinh(-1.0 / a_snr), snr_hi)

    # 4. noise map, log
    nz = np.where(finite, noise, np.nan)
    ln = np.log10(nz)
    p4 = render(np.nan_to_num(ln, nan=np.nanmax(ln)), 'cividis',
                np.nanpercentile(ln, 1), np.nanpercentile(ln, 99.5))

    # 5. PSF, log of peak-normalised. The FULL kernel from data/psf (with its wings) when
    # it exists on disk; the trimmed stamp kernel otherwise, padded to twice its width so
    # the core does not fill the panel as giant blocks.
    k = (psf_full if psf_full is not None else psf)
    k = k / np.nanmax(k)
    if psf_full is None:
        k = np.pad(k, k.shape[0] // 2)
    lk = np.log10(np.clip(k, 1e-4, None))
    p5 = render(lk, 'inferno', -4, 0)

    sprite = np.concatenate([p1, p2, p3, p4, p5], axis=1)
    # card thumbnail: central half of the image panel
    c0 = PANEL // 4
    thumb = Image.fromarray(p1[c0:PANEL - c0, c0:PANEL - c0]).resize((THUMB, THUMB), Image.LANCZOS)
    return sprite, thumb


def jpeg_bytes(im, q):
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q, optimize=True, progressive=True)
    return buf.getvalue()


def human(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024 or unit == 'GB':
            return f'{nbytes:.0f} {unit}' if unit == 'B' else f'{nbytes:.1f} {unit}'
        nbytes /= 1024


# ─────────────────────────────── catalogue ───────────────────────────────

def pretty_psf(p, scale):
    if not p:
        return None
    m = p.get('method', '')
    names = {
        'empirical': 'Empirical, from field stars',
        'empirical_median': 'Median of other fields’ empirical PSFs',
        'empirical_donor': 'Empirical, borrowed from a field observed the same day',
        'inject_wfpc2_psfdb': 'Model (MAST PSF database), re-drizzled like the data',
        'inject_stdpsf': 'Model (STScI STDPSF), re-drizzled like the data',
        'inject_acs_fdpsf': 'Model (ACS focus-diverse ePSF), re-drizzled like the data',
        'model_stpsf': 'Model (STPSF)',
    }
    label = names.get(m) or ('Model, analytically broadened' if m.startswith('broadened') else m)
    fwhm = p.get('fwhm_arcsec') or (p.get('fwhm_pix') * scale if p.get('fwhm_pix') else None)
    return dict(method=m, label=label, n_stars=p.get('n_stars'),
                fwhm=round(fwhm, 3) if fwhm else None,
                err=round(p['psf_err_frac'], 4) if p.get('psf_err_frac') is not None else None,
                err_lower_bound=bool(p.get('err_lower_bound')),
                kernel=p.get('trimmed_kernel_size'))


def pretty_mask_source(src):
    if not src:
        return None
    if src == 'drawn':
        return 'Drawn by hand'
    m = re.match(r'(accepted|edited|reprojected)_from_(\w+)', src)
    if m:
        verb = {'accepted': 'Accepted after review from', 'edited': 'Reviewed and edited from',
                'reprojected': 'Reprojected from'}[m.group(1)]
        return f'{verb} the {band_label(m.group(2))} mask'
    return src


def empty_reason(s):
    """The positions provenance's deliberately_empty text, without the bookkeeping."""
    if not s:
        return None
    s = re.sub(r';?\s*empty Grid2DIrregular written.*$', '', s)
    s = re.sub(r'\s*\(\d{4}-\d{2}-\d{2}\)', '', s)
    s = re.sub(r'^deliberately left without image positions:?\s*', '', s).strip()
    if not s:
        return 'The images were too uncertain to mark, or the counter-images are below the noise.'
    s = re.sub(r'^arc only:\s*', 'Only an arc is seen. ', s)
    return s[0].upper() + s[1:] + ('' if s.endswith('.') else '.')


def notes_of(rec):
    out = []
    if not isinstance(rec, dict):
        return out
    for key in ('note', 'edited', 'CORRECTION'):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict):
            txt = v.get('note') or v.get('reason') or v.get('what')
            if isinstance(txt, str):
                out.append(txt.strip())
    return out


def clean_reason(r):
    """A category reason for the page: no attribution, and none when it only restates the category."""
    r = re.sub(r'\s*\((?:Sam|by eye)[^)]*\)', '', r or '').strip(' .;')
    return None if r.lower() in ('', 'complex', 'complex (sam)') else r[0].upper() + r[1:]


def read_position_sets(path):
    """[[(y, x), ...], ...] arcsec: one set per source. A one-set file is a single
    Grid2DIrregular, a multi-set file autoconf's list type (positions_io.py in the HST repo)."""
    d = json.load(open(path))
    grids = d['values'] if d.get('type') == 'list' else [d]
    out = []
    for g in grids:
        v = g['arguments']['values']
        vals = v.get('array', v.get('values', [])) if isinstance(v, dict) else v
        out.append([(float(y), float(x)) for y, x in vals])
    return out


def build(out, only=None):
    os.makedirs(os.path.join(out, 'img'), exist_ok=True)
    cat = dict(telescopes=[], samples=[], lenses=[])
    for tel in TELESCOPES:
        repo = tel['path']
        files = tracked(repo)
        zarch = load_zenodo_archives(repo) if tel['key'] == 'hst' else {}
        samples = load(repo, 'samples/lens_samples.json')
        einstein = load(repo, 'catalogues/lens_einstein_radii.json')
        inst_j = load(repo, 'provenance/lens_instrument.json')
        exp_j = load(repo, 'provenance/lens_exptime.json')
        psf_j = load(repo, 'provenance/lens_psf.json')
        mask_j = load(repo, 'provenance/lens_masks.json')
        pos_j = load(repo, 'provenance/lens_positions.json')
        tel_entry = dict(key=tel['key'], name=tel['name'], short=tel['short'],
                         repo=f"https://github.com/{tel['gh']}", samples=[])
        for sample, sblock in samples.items():
            stamp_dirs = sorted({os.path.dirname(f) for f in files
                                 if f.startswith(f'data/stamps/{sample}/')
                                 and re.search(r'/stamp[\d.]+_sci\.fits$', f)})
            # the lens and band are the last two parts at either depth
            lenses = sorted({d.split('/')[-2] for d in stamp_dirs})
            if not lenses:
                continue
            title, blurb, refs = SAMPLES.get(sample, (sample, sblock.get('description', ''), ''))
            s_entry = dict(key=sample, telescope=tel['key'], title=title, blurb=blurb, refs=refs,
                           lenses=[], bands=set(), categories={})
            dl = {}      # filter -> [n bands, bytes]: the per-(sample, filter) release archives
            for lens in lenses:
                if only and lens != only:
                    continue
                entry = sblock['lenses'].get(lens, {}) if isinstance(sblock['lenses'], dict) else {}
                L = dict(name=lens, token=token(lens), sample=sample, telescope=tel['key'],
                         note=entry.get('note'), bands=[],
                         band_mode=entry.get('band_mode'), category=entry.get('category'),
                         category_reason=clean_reason(entry.get('category_reason')),
                         gold=bool(entry.get('etherington_gold')))
                er = einstein.get(lens, {})
                L['zl'] = er.get('zlens', entry.get('redshift_lens'))
                L['zs'] = er.get('zsrc', entry.get('redshift_source'))
                L['sigma'] = er.get('sigma_kms')
                # provenance for the catalogue numbers (the linked citation the page shows
                # under each): the row's own per-field _src where set (e.g. J1419+5117's
                # DESI lens redshift), else the row-level source short-formed.
                L['zl_src'] = cite_obj(er, 'zlens_src')
                L['zs_src'] = cite_obj(er, 'zsrc_src')
                L['sigma_src'] = cite_obj(er, 'sigma_src')
                L['zl_kind'] = er.get('zlens_kind')
                L['zs_kind'] = er.get('zsrc_kind')
                # reference-only photometric z (J0223-0651): only where the row carries one
                L['zs_photoz'] = None
                if isinstance(er.get('zsrc_photoz_ref'), dict):
                    L['zs_photoz'] = cite_obj(er, 'zsrc_photoz_ref')
                    L['zs_photoz']['value'] = er['zsrc_photoz_ref'].get('value')
                te = er.get('theta_e_arcsec')
                L['theta_e'] = te
                L['theta_e_src'] = cite_obj(er)
                if tel['key'] == 'jwst' and lens in JWST_THETA_E:
                    L['theta_e'], L['theta_e_src'] = JWST_THETA_E[lens]
                ldirs = {d.split('/')[-1]: d for d in stamp_dirs if d.split('/')[-2] == lens}
                bands = sorted(ldirs, key=lambda b: (PIVOT.get(base_band(b), 9), b))
                for band in bands:
                    bdir = ldirs[band]
                    bfiles = sorted(f for f in files if os.path.dirname(f) == bdir)
                    sci_rel = next(f for f in bfiles if re.search(r'/stamp[\d.]+_sci\.fits$', f))
                    prefix = os.path.basename(sci_rel)[:-len('_sci.fits')]
                    ab = lambda rel: os.path.join(repo, rel)
                    with fits.open(ab(sci_rel)) as h:
                        sci = h[0].data.astype(float)
                        hdr = h[0].header
                    noise_rel = f'{bdir}/{prefix}_noise.fits'
                    with fits.open(ab(noise_rel)) as h:
                        noise = h[0].data.astype(float)
                        nhdr = h[0].header
                    psf = fits.getdata(ab(f'{bdir}/psf.fits')).astype(float)
                    mrel = f'{bdir}/{prefix}_mask.fits'
                    mask = fits.getdata(ab(mrel)).astype(bool) if mrel in files else None
                    w = WCS(hdr)
                    scale = float(np.mean(proj_plane_pixel_scales(w.celestial)) * 3600)
                    n = sci.shape[0]
                    c = (n - 1) / 2
                    if 'ra' not in L:
                        sky = w.celestial.pixel_to_world(c, c)
                        L['ra'], L['dec'] = round(sky.ra.deg, 6), round(sky.dec.deg, 6)
                        L['radec'] = sky.to_string('hmsdms', sep=':', precision=2)
                    prel = f'{bdir}/{prefix}_positions.json'
                    sets = read_position_sets(ab(prel)) if prel in files else []
                    position_sets = [[(c + x / scale, c - y / scale) for y, x in st] for st in sets]
                    positions = [p for st in position_sets for p in st]
                    theta_px = (L['theta_e'] / scale) if L.get('theta_e') else None
                    # a group-centred (multi_galaxy) stamp: model and θ_E circle on the deflector
                    dpix = isophote_model.deflector_pixel(ab(sci_rel)) if tel['key'] == 'hst' else None
                    # the full PSF kernel (with wings) lives in the gitignored data/psf tree;
                    # use it when on disk, else the trimmed stamp kernel
                    full_rel = os.path.join(bdir.replace('data/stamps/', 'data/psf/', 1), 'psf_kernel.fits')
                    psf_full = fits.getdata(ab(full_rel)).astype(float) if os.path.isfile(ab(full_rel)) else None
                    sprite, thumb = panels(sci, noise, psf, mask, position_sets, theta_px, scale, dpix,
                                           psf_full=psf_full)
                    img_name = f"img/{L['token']}__{band}.jpg"
                    with open(os.path.join(out, img_name), 'wb') as f:
                        f.write(jpeg_bytes(Image.fromarray(sprite), 84))
                    instr = dig(inst_j, sample, lens, band)
                    prec = dig(pos_j, sample, lens, band) or {}
                    mrec = dig(mask_j, sample, lens, band) or {}
                    barchive = band_archive(zarch, sample, band) if zarch else None
                    B = dict(band=band, label=band_label(band), colour=band_colour(band),
                             pivot=PIVOT.get(base_band(band)),
                             instrument=INSTRUMENT.get(instr, instr), npix=n,
                             scale=round(scale, 4), size=round(n * scale, 2),
                             exptime=dig(exp_j, sample, lens, band),
                             unit=hdr.get('BUNIT'), noisecor=nhdr.get('NOISECOR'),
                             psf=pretty_psf(dig(psf_j, sample, lens, band), scale),
                             psf_px=psf.shape[0],
                             mask=(dict(frac=round(float(mask.mean()), 3),
                                        source=pretty_mask_source(mrec.get('source')))
                                   if mask is not None else None),
                             positions=(dict(n=len(positions), sets=[len(st) for st in sets if st],
                                             empty=empty_reason(prec.get('deliberately_empty')))
                                        if prel in files else None),
                             notes=notes_of(mrec), img=img_name,
                             archive=barchive,
                             files=[dict(name=os.path.basename(f),
                                         size=human(os.path.getsize(ab(f))),
                                         **gh_link(tel, f))
                                    for f in bfiles])
                    L['bands'].append(B)
                    d = dl.setdefault(base_band(band), [0, 0])
                    d[0] += 1
                    d[1] += sum(os.path.getsize(ab(f)) for f in bfiles)
                    s_entry['bands'].add(base_band(band))
                    if 'thumb' not in L or (CARD_BAND.index(base_band(band)) if base_band(band) in CARD_BAND else 99) < L['_tb']:
                        L['thumb'] = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes(thumb, 78)).decode()
                        L['_tb'] = CARD_BAND.index(base_band(band)) if base_band(band) in CARD_BAND else 99
                    print(f"  {tel['short']:4s} {sample:13s} {lens:12s} {band:9s} {n}px", flush=True)
                L.pop('_tb', None)
                if L['category']:
                    s_entry['categories'][L['category']] = s_entry['categories'].get(L['category'], 0) + 1
                cat['lenses'].append(L)
                s_entry['lenses'].append(L['token'])
            s_entry['bands'] = sorted(s_entry['bands'], key=lambda b: PIVOT.get(b, 9))
            s_entry['band_colours'] = {b: band_colour(b) for b in s_entry['bands']}

            def dl_url(filt):
                """The Zenodo archive covering this (sample, filter), as (name, row), or
                None for a telescope with no record yet. The per-filter zip is the smallest
                archive holding it; bells (one filter) has only bells_all.zip. filt is the
                uppercase base band ('F606W'); archive names keep the repo's case ('f606W')."""
                for name in (f"{sample}_f{filt[1:]}.zip", f'{sample}_all.zip'):
                    if name in zarch:
                        return name, zarch[name]
                return None, None

            s_entry['downloads'] = [dict(filter=f, bands=n, size=human(nb),
                                         archive=an, url=(row['url'] if row else None))
                                    for f, (n, nb) in sorted(dl.items(), key=lambda t: PIVOT.get(t[0], 9))
                                    for an, row in [dl_url(f)]]
            all_name, all_row = f'{sample}_all.zip', zarch.get(f'{sample}_all.zip')
            s_entry['download_all'] = dict(archive=all_name, url=all_row['url'],
                                           size=human(int(all_row['size_bytes']))) if all_row else None
            s_entry['download_total'] = human(sum(nb for _, nb in dl.values()))
            if s_entry['lenses']:
                cat['samples'].append(s_entry)
                tel_entry['samples'].append(sample)
        # Preferred display order: the flagship samples first (Sam, 2026-11-26), the rest
        # in their manifest order.
        tel_entry['samples'].sort(key=lambda s: (SAMPLE_ORDER.index(s) if s in SAMPLE_ORDER
                                                 else len(SAMPLE_ORDER)))
        cat['telescopes'].append(tel_entry)

    html = open(os.path.join(HERE, 'template.html')).read()
    html = html.replace('/*__CATALOGUE__*/null', json.dumps(cat, separators=(',', ':')))
    with open(os.path.join(out, 'index.html'), 'w') as f:
        f.write(html)
    nb = sum(len(L['bands']) for L in cat['lenses'])
    print(f"wrote {out}/index.html ({os.path.getsize(os.path.join(out, 'index.html')) / 1e6:.1f} MB), "
          f"{len(cat['lenses'])} lenses, {nb} bands")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=os.path.join(HERE, '_site'))
    ap.add_argument('--only', help='build one lens (for testing)')
    args = ap.parse_args()
    build(args.out, args.only)
