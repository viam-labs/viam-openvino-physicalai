import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_build_packages_module_with_exec_bits():
    subprocess.run(["bash", "build.sh"], cwd=ROOT, check=True)
    with tarfile.open(ROOT / "module.tar.gz") as tar:
        members = {m.name.removeprefix("./"): m for m in tar.getmembers() if m.isfile()}
    assert set(members) == {
        "meta.json", "requirements.txt", "run.sh", "setup.sh",
        "src/__init__.py", "src/main.py", "src/policy.py", "src/wire.py",
    }
    assert members["run.sh"].mode & 0o111 and members["setup.sh"].mode & 0o111
    assert not (ROOT / "run.sh").stat().st_mode & 0o111  # the source tree stays non-executable
