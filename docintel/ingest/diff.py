from dataclasses import dataclass

@dataclass
class CorpusDiff:
    to_add: list[str]       # filenames new to disk
    to_update: list[str]    # filenames with changed hash
    to_delete: list[str]    # filenames gone from disk
    unchanged: list[str]    # filenames with same hash (no-op)

def compute_diff(
    manifest: dict[str, str],
    current: dict[str, str],
) -> CorpusDiff:
    prev_names = set(manifest)
    curr_names = set(current)

    to_add    = sorted(curr_names - prev_names)
    to_delete = sorted(prev_names - curr_names)
    to_update, unchanged = [], []
    for name in sorted(curr_names & prev_names):
        if current[name] != manifest[name]:
            to_update.append(name)
        else:
            unchanged.append(name)

    return CorpusDiff(to_add, to_update, to_delete, unchanged)

if __name__ == "__main__":
    # Quick self-test with synthetic data.
    m = {"a.pdf": "aaa", "b.pdf": "bbb", "old.pdf": "ooo"}
    c = {"a.pdf": "aaa", "b.pdf": "BBB", "new.pdf": "nnn"}
    d = compute_diff(m, c)
    assert d.to_add == ["new.pdf"]
    assert d.to_update == ["b.pdf"]
    assert d.to_delete == ["old.pdf"]
    assert d.unchanged == ["a.pdf"]
    print("✓ diff engine self-test passed")
    print(f"  add={d.to_add} update={d.to_update} delete={d.to_delete} unchanged={d.unchanged}")