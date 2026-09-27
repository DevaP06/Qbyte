"""One-off patch (apply AFTER the v2.3 run finishes): adds --extra-tag to stack2.py
so a partially-scored CE (e5-base, uncertain pairs only) enters the stacker as an
extra feature, NaN where it did not score."""
import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "src" / "stack2.py"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep('''    if "coh_cos" in d.columns:
        cols += ["coh_cos", "coh_other_source"]''', '''    if "coh_cos" in d.columns:
        cols += ["coh_cos", "coh_other_source"]
    if "ce_x" in d.columns:  # partially scored extra CE (e5-base): NaN outside its band
        d["lx"] = _logit(d["ce_x"].to_numpy())
        d["x_vs_ce"] = d["lx"] - d["lc"]
        cols += ["lx", "x_vs_ce"]''')
rep('''def _suffix(tags: list, bag: int, coherence: bool = False) -> str:
    return "".join(tags) + (f"_bag{bag}" if bag > 1 else "") + ("_coh" if coherence else "")''',
    '''def _suffix(tags: list, bag: int, coherence: bool = False, extra: str = "") -> str:
    return "".join(tags) + (f"_bag{bag}" if bag > 1 else "") + ("_coh" if coherence else "") + extra''')
rep('''def _scores(kind: str, tags: list) -> pd.DataFrame:
    base = pd.read_parquet(CE_DIR / f"{kind}_scores{tags[-1]}.parquet")''', '''def _scores(kind: str, tags: list, extra: str = "") -> pd.DataFrame:
    base = pd.read_parquet(CE_DIR / f"{kind}_scores{tags[-1]}.parquet")
    if extra:
        x = pd.read_parquet(CE_DIR / f"{kind}_scores{extra}.parquet", columns=KEY + ["ce"]).rename(columns={"ce": "ce_x"})
        base = base.merge(x, on=KEY, how="left")''')
rep('''def _val_frame(val_run: str, tags: list, coherence: bool = False) -> pd.DataFrame:
    d = _attach_features(_scores("val", tags), config.FEATURES_DIR / f"train_{UNION}")''',
    '''def _val_frame(val_run: str, tags: list, coherence: bool = False, extra: str = "") -> pd.DataFrame:
    d = _attach_features(_scores("val", tags, extra), config.FEATURES_DIR / f"train_{UNION}")''')
rep('''def cv(val_run: str, tags: list, bag: int, coherence: bool = False) -> None:
    _stage(f"loading dev_val pairs + features (CE {tags}, bag {bag}, coherence {coherence})")
    d = _val_frame(val_run, tags, coherence)''', '''def cv(val_run: str, tags: list, bag: int, coherence: bool = False, extra: str = "") -> None:
    _stage(f"loading dev_val pairs + features (CE {tags}, bag {bag}, coherence {coherence}, extra {extra or '-'})")
    d = _val_frame(val_run, tags, coherence, extra)''')
rep('''    (CE_DIR / f"stack2_cv{_suffix(tags, bag, coherence)}.json")''', '''    (CE_DIR / f"stack2_cv{_suffix(tags, bag, coherence, extra)}.json")''')
rep('''def apply(val_run: str, test_run: str, out_dir: str, tags: list, bag: int, threshold_shift: float = 0.0,
          coherence: bool = False) -> None:''', '''def apply(val_run: str, test_run: str, out_dir: str, tags: list, bag: int, threshold_shift: float = 0.0,
          coherence: bool = False, extra: str = "") -> None:''')
rep('''    d = _val_frame(val_run, tags, coherence)
    boosters''', '''    d = _val_frame(val_run, tags, coherence, extra)
    boosters''')
rep('''    tce = _attach_features(_scores("test", tags), config.FEATURES_DIR / f"test_{UNION}")''',
    '''    tce = _attach_features(_scores("test", tags, extra), config.FEATURES_DIR / f"test_{UNION}")''')
rep('''    sfx = _suffix(tags, bag, coherence)''', '''    sfx = _suffix(tags, bag, coherence, extra)''')
rep('''"bag": bag, "coherence": coherence,''', '''"bag": bag, "coherence": coherence, "extra_tag": extra,''')
rep('''    parser.add_argument("--coherence", action="store_true",''', '''    parser.add_argument("--extra-tag", default="", help="Partially scored extra CE (e.g. '_base'), NaN where unscored.")
    parser.add_argument("--coherence", action="store_true",''')
rep('''        cv(args.val_run, _tags(args.tag), args.bag, args.coherence)
    else:
        apply(args.val_run, args.test_run, args.out_dir, _tags(args.tag), args.bag, args.threshold_shift, args.coherence)''',
    '''        cv(args.val_run, _tags(args.tag), args.bag, args.coherence, args.extra_tag)
    else:
        apply(args.val_run, args.test_run, args.out_dir, _tags(args.tag), args.bag, args.threshold_shift, args.coherence, args.extra_tag)''')
p.write_text(s, encoding="utf-8")
print("stack2.py: --extra-tag added")
