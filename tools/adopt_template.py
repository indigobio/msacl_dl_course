#!/usr/bin/env python3
"""Splice the course template's theme + slide master + layouts into decks.

PPTX-first toolchain (course_plan/DECISIONS.md 2026-09-08): slides/pptx/ decks
are the source of truth and are edited directly in PowerPoint. This tool gives
every deck the course master from slides/template/msacl_ds301.pptx, so that
"New Slide" offers on-style layouts and the theme colors show in every color
picker. Existing slides keep their shapes exactly as they are; only their
layout reference moves to the course master (matching layout by name, falling
back to Blank).

Run it again after editing the template to propagate a master/layout/theme
change to every deck. It is idempotent: an already-adopted deck gets its
course master replaced by the current template version.

Usage:
    python tools/adopt_template.py --all             # every deck in slides/pptx/
    python tools/adopt_template.py slides/pptx/lecture05_cnns.pptx

A deck with a live PowerPoint lock file (~$name.pptx) is skipped: close the
file in PowerPoint first, then re-run.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parent.parent
PPTX_DIR = ROOT / "slides" / "pptx"
TEMPLATE = ROOT / "slides" / "template" / "msacl_ds301.pptx"
MASTER_NAME = "MSACL DS301"

# old Office layout name -> course layout name (identity for already-adopted)
NAME_MAP = {
    "Title Slide": "Title Slide",
    "Title and Content": "Title and Body",
    "Two Content": "Two Column",
    "Picture with Caption": "Screenshot + Caption",
    "Title Only": "Title Only",
    "Blank": "Blank",
    "Title and Body": "Title and Body",
    "Two Column": "Two Column",
    "Screenshot + Caption": "Screenshot + Caption",
}


def _next_index(pkg, prefix, suffix=".xml"):
    """Next free N for partnames like /ppt/slideMasters/slideMasterN.xml."""
    n = 0
    for part in pkg.iter_parts():
        s = str(part.partname)
        if s.startswith(prefix) and s.endswith(suffix):
            try:
                n = max(n, int(s[len(prefix):-len(suffix)]))
            except ValueError:
                pass
    return n + 1


def _sorted_rels(part):
    return sorted(part.rels.values(), key=lambda r: int(r.rId[3:]))


def import_master(deck_prs, tmpl_prs):
    """Copy the template's master, layouts and theme into the deck package.
    Returns the new master part. Blobs are copied verbatim, so the rels of
    each new part must reproduce the source rIds exactly; the template's rels
    are normalized (contiguous from rId1) by make_template.py to guarantee it."""
    pkg = deck_prs.part.package
    src_master = tmpl_prs.slide_masters[0].part
    src_theme = src_master.part_related_by(RT.THEME)

    m_i = _next_index(pkg, "/ppt/slideMasters/slideMaster")
    t_i = _next_index(pkg, "/ppt/theme/theme")
    l_i = _next_index(pkg, "/ppt/slideLayouts/slideLayout")

    new_theme = Part(PackURI(f"/ppt/theme/theme{t_i}.xml"),
                     src_theme.content_type, pkg, src_theme.blob)
    new_master = Part(PackURI(f"/ppt/slideMasters/slideMaster{m_i}.xml"),
                      src_master.content_type, pkg, src_master.blob)

    new_layouts = {}  # src partname -> new part
    for rel in _sorted_rels(src_master):
        if rel.reltype == RT.SLIDE_LAYOUT:
            src_layout = rel.target_part
            new_layout = Part(PackURI(f"/ppt/slideLayouts/slideLayout{l_i}.xml"),
                              src_layout.content_type, pkg, src_layout.blob)
            l_i += 1
            new_layouts[str(src_layout.partname)] = new_layout

    # master rels, in source order so assigned rIds match the blob's r:id refs
    for rel in _sorted_rels(src_master):
        if rel.reltype == RT.SLIDE_LAYOUT:
            target = new_layouts[str(rel.target_part.partname)]
        elif rel.reltype == RT.THEME:
            target = new_theme
        else:
            raise ValueError(f"unexpected master rel {rel.reltype}")
        got = new_master.rels.get_or_add(rel.reltype, target)
        assert got == rel.rId, f"master rel drift: {got} != {rel.rId}"

    # each layout points back at its master (and nothing else in our template)
    by_name = {}
    for rel in _sorted_rels(src_master):
        if rel.reltype != RT.SLIDE_LAYOUT:
            continue
        src_layout = rel.target_part
        new_layout = new_layouts[str(src_layout.partname)]
        for lrel in _sorted_rels(src_layout):
            if lrel.reltype != RT.SLIDE_MASTER:
                raise ValueError(f"unexpected layout rel {lrel.reltype}")
            got = new_layout.rels.get_or_add(RT.SLIDE_MASTER, new_master)
            assert got == lrel.rId, f"layout rel drift: {got} != {lrel.rId}"
        by_name[src_layout.slide_layout.name] = new_layout

    return new_master, by_name, new_theme


def adopt(deck_path: Path) -> bool:
    lock = deck_path.parent / f"~${deck_path.name}"
    if lock.exists():
        print(f"SKIP {deck_path.name}: open in PowerPoint (lock file present) "
              f"— close it and re-run")
        return False

    prs = Presentation(str(deck_path))
    before = [(len(s.shapes), s.has_notes_slide) for s in prs.slides]

    tmpl = Presentation(str(TEMPLATE))
    new_master_part, by_name, new_theme = import_master(prs, tmpl)
    tmpl_master_layout_ids = [
        int(e.get("id"))
        for e in tmpl.slide_masters[0].element.find(qn("p:sldLayoutIdLst"))
    ]

    # hook the new master into the presentation part. Master ids and layout
    # ids share ONE uniqueness space (ECMA-376), and the imported master blob
    # carries the template's layout ids verbatim — the new master id must
    # clear those too, or PowerPoint flags the deck as corrupt.
    pres_part = prs.part
    id_lst = pres_part._element.find(qn("p:sldMasterIdLst"))
    old_entries = list(id_lst)
    used = {int(e.get("id")) for e in old_entries}
    for lid in tmpl_master_layout_ids:
        used.add(lid)
    rId = pres_part.relate_to(new_master_part, RT.SLIDE_MASTER)
    entry = parse_xml(
        f'<p:sldMasterId {nsdecls("p", "r")} id="{max(used | {2147483647}) + 1}" '
        f'r:id="{rId}"/>'
    )
    id_lst.append(entry)

    # re-target every slide to the same-named course layout (default: Blank)
    for slide in prs.slides:
        srel = next(r for r in slide.part.rels.values()
                    if r.reltype == RT.SLIDE_LAYOUT)
        old_name = srel.target_part.slide_layout.name
        new_layout_part = by_name[NAME_MAP.get(old_name, "Blank")]
        slide.part.rels.pop(srel.rId)
        slide.part.rels.get_or_add(RT.SLIDE_LAYOUT, new_layout_part)

    # the presentation part's own theme rel (default text styles) must follow
    # the course theme, or the old theme lingers in the package
    for r in list(pres_part.rels.values()):
        if r.reltype == RT.THEME:
            pres_part.rels.pop(r.rId)
    pres_part.rels.get_or_add(RT.THEME, new_theme)

    # drop every other master (the stock Office one, or a stale course master)
    for e in old_entries:
        e_rId = e.get(qn("r:id"))
        id_lst.remove(e)
        pres_part.rels.pop(e_rId)

    prs.save(str(deck_path))
    verify(deck_path, before)
    print(f"adopted {deck_path.name}: {len(before)} slides -> master "
          f"'{MASTER_NAME}' ({len(by_name)} layouts)")
    return True


def verify(deck_path: Path, before):
    prs = Presentation(str(deck_path))
    assert len(prs.slide_masters) == 1, "expected exactly one master"
    master = prs.slide_masters[0]
    assert master.element.find(qn("p:cSld")).get("name") == MASTER_NAME
    after = [(len(s.shapes), s.has_notes_slide) for s in prs.slides]
    assert after == before, "slide content changed during adoption"
    names = {l.name for l in master.slide_layouts}
    for s in prs.slides:
        assert s.slide_layout.name in names
    # master ids and layout ids share one id space (PowerPoint enforces this)
    m_ids = {e.get("id") for e in
             prs.part._element.find(qn("p:sldMasterIdLst"))}
    l_ids = {e.get("id") for e in
             master.element.find(qn("p:sldLayoutIdLst"))}
    assert not (m_ids & l_ids), f"master/layout id collision: {m_ids & l_ids}"


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if not TEMPLATE.exists():
        print("template missing — run tools/make_template.py first", file=sys.stderr)
        return 1
    targets = (sorted(PPTX_DIR.glob("lecture*.pptx")) if argv[1] == "--all"
               else [Path(a).resolve() for a in argv[1:]])
    status, done = 0, 0
    for deck in targets:
        try:
            done += adopt(deck)
        except Exception as exc:
            print(f"ERROR in {deck.name}: {exc}", file=sys.stderr)
            status = 1
    print(f"{done}/{len(targets)} deck(s) adopted")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
