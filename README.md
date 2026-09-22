# dxfgen

Generates original, parametric, machine-ready DXF files for CNC routers and
lasers, and packages them into folders you can sell.

Every shape here comes out of an equation or a boolean of primitives. Nothing
is traced, copied or reproduced from an existing design, and there is no image
input anywhere in the pipeline — originality is structural, not a promise.

```bash
pip install -e .

dxfgen list                                      # what can be made
dxfgen list trays                                # and what it takes
dxfgen make trays -p length=480 -p width=300     # one design
dxfgen bundle --niche trays --count 50 --seed 42 # fifty, zipped
dxfgen validate somebody-elses-file.dxf --tool 6 # will this cut?
```

---

## What makes a file machine-ready

Most generated DXFs import with warnings. These do not, and the reason is that
the validator refuses to export anything that fails, rather than exporting it
with a note. **An invalid design never reaches the disk.**

**Reachability, not corner radius.** A round cutter cannot enter a sharp inside
corner. The usual answer is "minimum inside radius ≥ tool radius", which is a
proxy, and a leaky one: it says nothing about a slot narrower than the cutter
or a feature the cutter cannot get into sideways. dxfgen instead runs a
morphological **opening** — erode by the tool radius, dilate back — and
compares it to the original region. What survives is exactly what the cutter
can reach. That single test subsumes minimum radius, minimum feature width and
minimum slot width, and it reports the *measured residual* in millimetres
rather than a pass/fail.

The dual test, **closing**, does the same for concave features on an outer
profile, where the cutter works around the part instead of inside a pocket.

**Corners are relieved by construction.** Dogbone and T-bone relief is
booleaned into the contour, not exported as overlapping circles that CAM flags
as duplicate vectors. Rounded corners are produced by the opening itself, so
they are machinable by definition — unlike vertex filleting, which silently
clamps the radius to half the shortest adjacent edge and fails outright on a
tessellated curve.

Measured: a relieved corner leaves **0.049 mm** of material against a 6.35 mm
cutter. The same corner left sharp leaves **1.07 mm**.

**Kerf is compensated per part, not per feature.** On a laser, outlines grow by
half a kerf and cutouts shrink by half a kerf. Compensating only the slot
leaves the joint a full kerf loose, because the beam also takes half a kerf off
the tab.

Measured on a 3 mm joint with a 0.15 mm kerf and 0.2 mm clearance asked for.
A 3.200 mm mortise is drawn at **3.050** so the beam opens it back to 3.200; a
24.000 mm tab is drawn at **24.150** so the beam narrows it back to 24.000. The
3 mm material then enters a 3.200 mm slot: exactly the 0.200 mm asked for.

**A design that assembles knows how.** A nest of six flat rectangles is what
the machine needs and the worst possible picture of a bookcase. Boxes, docks
and furniture record where each panel sits in the finished object, and that is
drawn in isometric projection — for the per-design image and for the contact
sheet tile, so a bundle of shelf units shows shelf units rather than cut
sheets. It is 2D line work: panels sorted back to front by centroid and
painted, which is exact for the orthogonal, non-interpenetrating assemblies
this supports. Flat products carry no assembly, because for them the cut file
already is the picture.

Putting the drawing and the joinery on the same numbers is what makes it worth
having. The upright's mortise heights and the drawn shelf heights come from
one function, so a shelf cannot be drawn where its mortise was not cut — and
the first contact sheet drawn this way immediately showed a 1560 mm bookcase
with two shelves 1400 mm apart, a flaw that had been invisible for as long as
the output was a nest of panels.

**A wedged tenon needs somewhere to pull from.** A flat-pack shelf held by
friction alone racks and works loose. Its tenons can instead stand proud of
the upright and take a tapered wedge, which is the classic knock-down joint
and tightens every time it is tapped. The mechanism is one number: the wedge
slot reaches back *inside* the upright's outer face by the bite, so the
wedge's straight side can bear on the upright while its taper pushes the far
end of the slot outwards, pulling the shoulder tight. A slot flush with the
upright looks identical and does nothing.

**A recess has to follow the outline it is set into.** A tray is its shape
moved inwards, and on an outline with small lobes — a paw's toes — offsetting
eats them and rounding blobs what is left, so the result reads as a puddle
inside a paw. That is caught by comparing how convoluted each is relative to
its own size (perimeter² / area, which is dimensionless): measured across
every shape, the paw comes out 1.40× its outline and everything that trays
well sits between 0.78 and 1.05. So the check names no shape, and would catch
the next one.

**Parameters are drawn against each other, not independently.** A sampler
that picks a tab width and a panel width separately proposes a 30 mm tab for a
76 mm panel; one that picks a shelf count and a height separately gives a
1560 mm bookcase two shelves. Every such variant is correctly refused, and a
quarter of the work is thrown away to find that out. Each sampler now draws
the containing thing first and sizes what goes in it by solving the same
inequality the geometry checks — never by restating it, so the two cannot
drift apart. Measured across nine niches and seven seeds, 3% of proposals are
rejected, and a contract test holds each niche under 12%.

**Files say what machine they are for.** A DXF has nowhere standard to record
this, so `dxfgen validate` on a laser file would otherwise apply router rules
and flag every finger joint. dxfgen writes the mode, cutter, kerf and thickness
into header custom variables. A file that does not carry them, with no `--tool`
given, has the cutter checks skipped and says so — it is not judged against a
guess.

---

## Niches

| niche | what it makes | machine |
|---|---|---|
| `trays` | Serving, valet and snack trays with recessed compartments and handles | router |
| `boards` | Cutting and charcuterie boards with juice grooves, hand holds and hang holes | router |
| `coasters` | Coaster sets in five shapes with engraved patterns and a holder | router |
| `ashtrays` | Cigar and whiskey trays with a glass recess, cigar rests and an ash well | router |
| `stands` | Three-part slot-together phone and tablet docks in four back profiles, no glue | router |
| `boxes` | Finger-jointed boxes, plain or with cross-lapped compartments, tabbed floor, optional lid | laser |
| `furniture` | Flat-pack shelves with wedged or friction through-tenons, and cross-leg tables, nested onto sheet stock | router |
| `seasonal` | Themed plaques, shaped trays and coaster sets from parametric curves | either |
| `ornaments` | Ornaments and keychains from parametric shapes, nested as a set | laser |

Nine parametric shapes back the last two: heart, star, moon, pumpkin, ghost,
tree, paw, leaf, snowflake. Each is an equation or a boolean of primitives, and
each is relieved to whatever cutter it is asked about.

---

## Commands

### `dxfgen list [niche]`

Without an argument, the niches. With one, every parameter it accepts, its
default, its limits and what it means.

### `dxfgen make <niche> [-p key=value ...]`

One design, into `output/<niche>/<slug>/`:

| file | what it is |
|---|---|
| `<slug>.dxf` | DXF R2010, millimetres, closed LWPOLYLINEs, for CAM |
| `<slug>.svg` | SVG in millimetres, for LightBurn and friends |
| `<slug>_template.pdf` | 1:1 print template, tiled with alignment crosses if it does not fit one sheet |
| `<slug>_preview.png` | what the file contains, layer by layer |
| `<slug>_mockup.png` | the listing image: wood-grained, and drawn assembled where the design assembles |
| `<slug>_assembly.png` | what it looks like put together — boxes, docks and furniture only |
| `README.txt` | material, depths per layer, cutting order, licence summary |

```bash
dxfgen make boards -p length=420 -p width=280 -p juice_groove=true
dxfgen make boxes --preset laser-3mm-ply -p width=180 -p height=90
dxfgen make stands -p size=tablet -p cable_slot=true -o ~/cnc
```

Useful options: `--mode router|laser`, `--out DIR`, `--paper a4|a3|letter`,
`--format` (repeatable, to skip the slow outputs), `--preset NAME`.

### `dxfgen bundle`

A whole niche, ready to upload.

```bash
dxfgen bundle --niche trays --count 50 --seed 42 --seller "Northwood Digital"
dxfgen bundle --all --count 20
```

Adds to the niche folder:

- `CONTACT_SHEET.png` — every design on one page, captioned
- `LICENSE.txt` — sell what you make, do not pass on the files
- `INDEX.txt` — the manifest, the seed, and why the bundle is short if it is
- `<niche>_bundle.zip` — built from the manifest, so leftovers from an earlier
  run with a different seed are reported and excluded, never shipped

The same seed and count reproduce the same designs exactly, and raising the
count extends the set rather than reshuffling it: variant *i* depends only on
`(seed, i)`.

Bundles are split at 20 MB, because Etsy, Gumroad and most other download
marketplaces cap a single file there and a 25 MB zip cannot be uploaded at all.
Every part carries the licence, the index and the contact sheet, so a buyer
who has only one part still has the terms they are bound by, and a design
folder is never split across parts. `--max-zip-mb 0` gives one file of any
size.

### `dxfgen validate <file.dxf> ...`

Works on any DXF, not just ones from here. Checks structure (ezdxf audit, open
contours, splines, duplicates, self-intersections, units, origin) and, when it
knows the machine, whether the geometry is reachable.

```bash
dxfgen validate bought-file.dxf --tool 6.35
dxfgen validate bought-file.dxf --mode laser --kerf 0.12
```

Exit codes: `0` clean, `1` the file has errors, `2` the command was wrong.

---

## The layer standard

| layer | ACI | meaning |
|---|---|---|
| `CUT_OUTSIDE` | 1 red | outer profile, tool outside the line |
| `CUT_INSIDE` | 5 blue | interior through cuts, tool inside the line |
| `POCKET_<depth>` | 3 green + | cleared to that depth in mm, one layer per depth |
| `ENGRAVE` | 6 magenta | on the line, V-bit or engraving cutter |
| `DRILL` | 2 yellow | circle diameter is the hole diameter |
| `INFO` | 8 grey | annotation, never machine it |

All cut geometry is closed LWPOLYLINEs or circles. No splines, no duplicates,
no zero-length segments, no overlapping lines. `$INSUNITS = 4` (millimetres),
origin at the bottom-left of the bounding box. Engraving may legitimately be
open — a ray or a hatch line is a stroke, not a boundary — and is the one
exemption.

---

## Manufacturing rules

Enforced before export. Defaults, all configurable per design:

| rule | default |
|---|---|
| minimum wall between features | 8 mm |
| material left under a pocket | 5 mm |
| minimum feature width | 1.1 × tool diameter |
| maximum residual in a corner | 0.35 mm |
| joint slot width | thickness + 0.2 mm clearance |
| router cutter | 6.35 mm (¼ in) |
| laser kerf | 0.15 mm |
| sheet, furniture | 1220 × 2440 mm |
| sheet, laser | 600 × 400 mm |

Furniture parts are nested onto sheet stock with a first-fit-decreasing shelf
packer and labelled on `INFO`. It is not optimal — no practical nester is — but
it is stable, fast, and leaves a layout an operator can read.

Smaller sets — coasters, boxes — are scored differently, on **the board you
have to buy** rather than on bounding-box area. The two disagree: shelf packing
beats a grid by about 20% on bounding-box area for coaster sets, and is 16%
*worse* on panel area, because it lays parts in long rows and a long row forces
a wider board. Every grid width and the packer are tried and the cheapest board
wins. Four of ten coaster sets and six of twelve boxes come out on a smaller
board than a fixed grid gave them.

## Presets

`--preset` loads a YAML file of parameters; anything you pass with `-p` wins
over it.

| preset | for |
|---|---|
| `router-18mm-ply` | flat-pack work, 18 mm birch ply, ¼ in cutter |
| `router-19mm-hardwood` | boards and trays, ¾ in hardwood |
| `router-12mm-cutter` | a big cutter that will not reach small corners |
| `laser-3mm-ply` | 3 mm birch ply on a CO₂ laser |
| `laser-6mm-ply` | 6 mm ply, wider kerf |

They live in `dxfgen/configs/`. Copy one and edit it; `--preset ./mine.yaml`
takes a path.

---

## Using the files

**Kerf is a measurement, not a setting.** Cut a 20 mm test square, measure it,
and set `kerf` to `20 − measured`. The 0.15 mm default is a starting point for
3 mm ply on a 40–60 W CO₂ laser and will be wrong for your machine.

**A larger cutter will not fit a file cut for a smaller one.** Inside corners
are pre-relieved to the cutter named in the README. Run `dxfgen validate
file.dxf --tool <yours>` before cutting; it will tell you in millimetres what
will not clear.

**Print templates at 100%.** Turn off "fit to page". Check the 100 mm bar on
the sheet with a ruler before you cut anything.

**Check joint fit on an offcut first.** The 0.2 mm clearance suits plywood cut
dry. MDF, acrylic and damp stock all behave differently.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest          # 1067 tests
```

```
dxfgen/
  core/       geometry, layers, validation, DXF/SVG/PDF/PNG export
  niches/     one module per niche, plus the shared generator contract
  bundles/    per-design output folders, contact sheets, licence, zip
  configs/    YAML presets
  cli.py      list / make / bundle / validate
tests/        one module per source module
```

Adding a niche means writing geometry, not plumbing: subclass `Generator`,
declare a pydantic schema, implement `generate()` and `sample_params()`, and
register it. Seeded variants, validation, skip reporting, export, packaging and
the CLI all come from the base class.

---

## Known limitations

Stated plainly, because finding these out at the machine is expensive.

- **Mockups are not renders.** A flat product gets a wood-textured top view,
  where a recess reads as a tint rather than as depth. An assembled one gets
  the isometric with grain on it: no perspective, no shadows, no
  hidden-surface solver, and one grain field across the whole drawing rather
  than one per panel — so on an upright the grain runs in the projected
  direction rather than along the panel.
- **Assemblies must be orthogonal.** Panels lie in one of three planes at
  right angles. Anything mitred, hinged or curved cannot be described, and a
  generator that cannot describe itself simply carries no assembly.
- **Sheet nesting leaves real waste.** Measured over 12 furniture designs, part
  area against the area the packer actually occupied: median 73%, range 36–85%.
  A shelf packer leaves the gaps a shelf packer leaves, and the worst cases are
  designs whose parts differ most in height.
- **Only shelf units can be wedged.** A cross-lapped table is held by its own
  geometry, so there is nothing for a wedge to pull tight, and there is no cam
  or captive-nut hardware anywhere.
- **Boxes have no hinges or drawers.** A plain or compartmented open box with
  an optional lift-off lid is the whole range.
- **The pumpkin's lobes are engraved** rather than cut into the silhouette.
  That is the right call — modulating the profile makes it read as a flower —
  but it means the silhouette alone is a rounded rectangle.
- **PDF templates are verified geometrically**, by figure dimensions and
  MediaBox, not by rasterising the output. No PDF rasteriser was available in
  the build environment.
- **Cigar trays are one layout** in every variant: glass recess left, cigar
  rests right. That is close to what a cigar tray is, and inventing variety
  there is a design job rather than a fix.
- **A large seasonal snowflake is router-cuttable, a small one is not.** Below
  roughly 160 mm the arms are finer than a 6.35 mm cutter and the design is
  refused with advice rather than quietly rounded off.

## Roadmap (v2)

- **3D relief STL from heightmaps** — carved trays and relief panels. Out of
  scope here on purpose; it is a different pipeline, not an extra exporter.
- **Image-to-vector tracing.** Deliberately absent. Every shape in this project
  is generated, which is what makes the output original and safe to sell;
  tracing an input would put that guarantee in the user's hands instead.

## Licence

The generator is yours to use. The designs it produces are yours to sell.
`LICENSE.txt` in each bundle is a plain-language template for what you grant
your own buyers — commercial use of the physical items, no redistribution of
the files. Put your shop name in it with `--seller`. It is a starting point,
not legal advice.
