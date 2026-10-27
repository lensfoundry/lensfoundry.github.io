#!/usr/bin/env python
"""Build the lensprep lens-image browser: one static page over the HST and JWST reductions.

    uv run --project ../HST_lenses_data_reduction python build.py [--out _site] [--only LENS]

Reads ONLY files tracked in git in the two reduction repos (the stamps, masks, PSFs and
positions under data/stamps/, and the info/ JSONs), so the browser shows exactly what a clone
carries. Writes

    <out>/index.html            the page: template.html with the catalogue embedded as JSON
    <out>/img/<lens>__<band>.jpg one sprite per band: image | lensed light | noise | PSF

Panels are drawn North up, East left (FITS column to the right, row upwards), so a (col, row)
quoted in a review note counts from the bottom-left corner. The "lensed light" panel subtracts
the deflector's azimuthal median profile: a finding aid, never photometry.

Download links point at the files on GitHub until the Zenodo records exist; then the manifest
replaces them (release plan, step 5).
"""
import argparse
import base64
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
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
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
    'slacs_gold': ('SLACS Gold', 'The 38 SLACS lenses graded "Gold" by Etherington et al. (2022).',
                   'Bolton et al. 2008; Auger et al. 2009; Etherington et al. 2022'),
    'slacs_other': ('SLACS', 'The remaining SLACS lenses with a clean early-type deflector.',
                    'Bolton et al. 2008; Auger et al. 2009'),
    'gallery': ('BELLS GALLERY', 'Lensed Lyman-alpha emitters at z 2–3 behind BOSS galaxies, imaged with WFC3/UVIS.',
                'Shu et al. 2016'),
    'bells': ('BELLS', 'BOSS emission-line lenses imaged with ACS in F814W.',
              'Brownstein et al. 2012'),
    'sl2s_dinos_a': ('SL2S + Project Dinos', 'SL2S lenses with deep GO-17130 F475X and an older optical band.',
                     'Sonnenfeld et al. 2013; Sheu et al. 2025'),
    'sl2s_dinos_x': ('SL2S + Project Dinos, F475X', 'SL2S lenses whose only deep band is the GO-17130 F475X.',
                     'Sonnenfeld et al. 2013; Sheu et al. 2025'),
    'sl2s_wf': ('SL2S, WFPC2', 'SL2S lenses with WFPC2 F606W imaging.',
                'Sonnenfeld et al. 2013'),
    'sl2s_snap': ('SL2S snapshots', 'SL2S lenses with only the 720 s F475X + F600LP snapshots.',
                  'Sonnenfeld et al. 2013'),
    'b1938': ('JVAS B1938+666', 'A radio-loud Einstein ring with 21 dithers per filter, ~24 h each.',
              'GO 8281 (PI Vegetti)'),
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


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def token(name):
    return name.replace('+', 'p')


def band_label(band):
    m = re.match(r'([fF]\d+[A-Za-z]+)(?:_v(\d+))?$', band)
    base = m.group(1).upper() if m else band.upper()
    return base + (f' (visit {m.group(2)})' if m and m.group(2) else '')


def base_band(band):
    return band.split('_')[0].upper()


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


def radial_profile_subtract(img, exclude):
    n0, n1 = img.shape
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - (n0 - 1) / 2, xx - (n1 - 1) / 2).round().astype(int)
    good = np.isfinite(img) & ~exclude
    prof = np.zeros(r.max() + 1)
    for k in range(r.max() + 1):
        sel = good & (r == k)
        if sel.any():
            prof[k] = np.median(img[sel])
        elif k:
            prof[k] = prof[k - 1]
    return img - prof[r]


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


def panels(sci, noise, psf, mask, positions_px, theta_px):
    n = sci.shape[0]
    finite = np.isfinite(sci) & np.isfinite(noise) & (noise < 1e7)
    sci = np.where(finite, sci, np.nan)
    mask = mask if mask is not None else np.zeros_like(sci, bool)
    yy, xx = np.indices(sci.shape)
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)

    outer = finite & ~mask & (r > 0.35 * n)
    sky = np.nanmedian(sci[outer]) if outer.any() else np.nanmedian(sci)
    sig = mad_sigma(sci[outer] if outer.any() else sci)
    sig_s = sig * 0.45   # rough sigma after the 0.8 px smoothing

    # 1. image: asinh about the sky
    a = 2.0 * sig
    img = np.arcsinh((sci - sky) / a)
    core = finite & (r < 0.25 * n)
    hi = np.nanpercentile(img[core], 99.9) if core.any() else np.nanmax(img)
    p1 = render(np.nan_to_num(img, nan=-9), 'inferno', np.arcsinh(-1.5 * sig / a), hi)

    # 2. lensed light: deflector's azimuthal median removed, contaminant mask shaded
    sub = radial_profile_subtract(sci - sky, mask | ~finite)
    sub = gaussian_filter(np.nan_to_num(sub), 0.8)   # light smoothing: a display aid for faint arcs
    b = 1.5 * sig_s
    sv = np.arcsinh(sub / b)
    reach = min(0.3 * n, 3.0 * theta_px) if theta_px else 0.3 * n
    win = finite & ~mask & (r < reach) & (r > 2)
    hi2 = np.nanpercentile(sv[win], 99.7) if win.sum() > 20 else np.nanmax(sv)
    hi2 = max(hi2, np.arcsinh(6))

    def over(ax):
        if mask.any():
            shade = np.zeros(mask.shape + (4,))
            shade[mask] = (0.0, 0.0, 0.0, 0.58)
            ax.imshow(shade, origin='lower', interpolation='nearest')
            cs = ax.contour(mask.astype(float), levels=[0.5], colors='white', linewidths=1.1)
            for c in getattr(cs, 'collections', [cs]):
                c.set_path_effects(HALO)
        if theta_px:
            circ = plt.Circle(((n - 1) / 2, (n - 1) / 2), theta_px, fill=False, color='white',
                              lw=1.1, ls=(0, (4, 3)), alpha=0.85)
            circ.set_path_effects(HALO)
            ax.add_patch(circ)
        for col, row in positions_px:
            ax.plot([col], [row], marker='+', ms=17, mew=2.0, color='#6ff0ff',
                    path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
    p2 = render(np.nan_to_num(sv, nan=0), 'inferno', np.arcsinh(-0.6 / 1.5), hi2, over)

    # 3. noise map, log
    nz = np.where(finite, noise, np.nan)
    ln = np.log10(nz)
    p3 = render(np.nan_to_num(ln, nan=np.nanmax(ln)), 'cividis',
                np.nanpercentile(ln, 1), np.nanpercentile(ln, 99.5))

    # 4. PSF, log of peak-normalised
    k = psf / np.nanmax(psf)
    lk = np.log10(np.clip(k, 1e-4, None))
    p4 = render(lk, 'inferno', -4, 0)

    sprite = np.concatenate([p1, p2, p3, p4], axis=1)
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


def build(out, only=None):
    os.makedirs(os.path.join(out, 'img'), exist_ok=True)
    cat = dict(telescopes=[], samples=[], lenses=[])
    for tel in TELESCOPES:
        repo = tel['path']
        files = tracked(repo)
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
            lenses = sorted({d.split('/')[3] for d in stamp_dirs})
            if not lenses:
                continue
            title, blurb, refs = SAMPLES.get(sample, (sample, sblock.get('description', ''), ''))
            s_entry = dict(key=sample, telescope=tel['key'], title=title, blurb=blurb, refs=refs,
                           lenses=[], bands=set())
            for lens in lenses:
                if only and lens != only:
                    continue
                entry = sblock['lenses'].get(lens, {}) if isinstance(sblock['lenses'], dict) else {}
                L = dict(name=lens, token=token(lens), sample=sample, telescope=tel['key'],
                         note=entry.get('note'), bands=[])
                er = einstein.get(lens, {})
                L['zl'] = er.get('zlens', entry.get('redshift_lens'))
                L['zs'] = er.get('zsrc', entry.get('redshift_source'))
                L['sigma'] = er.get('sigma_kms')
                te = er.get('theta_e_arcsec')
                L['theta_e'] = te
                L['theta_e_src'] = (er.get('source') or '').split(',')[0] or None
                if tel['key'] == 'jwst' and lens in JWST_THETA_E:
                    L['theta_e'], L['theta_e_src'] = JWST_THETA_E[lens]
                bands = sorted([d.split('/')[4] for d in stamp_dirs if d.split('/')[3] == lens],
                               key=lambda b: (PIVOT.get(base_band(b), 9), b))
                for band in bands:
                    bdir = f'data/stamps/{sample}/{lens}/{band}'
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
                    positions = []
                    if prel in files:
                        v = json.load(open(ab(prel)))['arguments']['values']
                        vals = v.get('array', v.get('values', []))   # an empty grid may be saved as a list
                        positions = [(c + x / scale, c - y / scale) for y, x in vals]
                    theta_px = (L['theta_e'] / scale) if L.get('theta_e') else None
                    sprite, thumb = panels(sci, noise, psf, mask, positions, theta_px)
                    img_name = f"img/{L['token']}__{band}.jpg"
                    with open(os.path.join(out, img_name), 'wb') as f:
                        f.write(jpeg_bytes(Image.fromarray(sprite), 84))
                    instr = dig(inst_j, sample, lens, band)
                    prec = dig(pos_j, sample, lens, band) or {}
                    mrec = dig(mask_j, sample, lens, band) or {}
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
                             positions=(dict(n=len(positions),
                                             empty=empty_reason(prec.get('deliberately_empty')))
                                        if prel in files else None),
                             notes=notes_of(mrec), img=img_name,
                             files=[dict(name=os.path.basename(f),
                                         size=human(os.path.getsize(ab(f))),
                                         url=f"https://github.com/{tel['gh']}/raw/main/{quote(f)}")
                                    for f in bfiles])
                    L['bands'].append(B)
                    s_entry['bands'].add(base_band(band))
                    if 'thumb' not in L or (CARD_BAND.index(base_band(band)) if base_band(band) in CARD_BAND else 99) < L['_tb']:
                        L['thumb'] = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes(thumb, 78)).decode()
                        L['_tb'] = CARD_BAND.index(base_band(band)) if base_band(band) in CARD_BAND else 99
                    print(f"  {tel['short']:4s} {sample:13s} {lens:12s} {band:9s} {n}px", flush=True)
                L.pop('_tb', None)
                cat['lenses'].append(L)
                s_entry['lenses'].append(L['token'])
            s_entry['bands'] = sorted(s_entry['bands'], key=lambda b: PIVOT.get(b, 9))
            s_entry['band_colours'] = {b: band_colour(b) for b in s_entry['bands']}
            if s_entry['lenses']:
                cat['samples'].append(s_entry)
                tel_entry['samples'].append(sample)
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
