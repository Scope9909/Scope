"""Stable, date-free Scope verification commands. Does not train or evaluate Test."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
COMMANDS={
    'preflight':'tools/preflight.py',
    'check-models':'tools/check_models.py',
    'test-codec':'contrib3_index_codec/test_adaptive_lossless_state_codec.py',
    'test-lowrank':'test_factorized_lite_predictor.py',
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=COMMANDS)
    args=parser.parse_args()
    script = ROOT / COMMANDS[args.command]
    # Embedded Python's ._pth can omit the script directory from sys.path.
    # Set it explicitly in the child process so sibling imports work without
    # changing the user's interpreter or the experiment modules.
    runner = (
        'import runpy,sys; '
        'from pathlib import Path; '
        'script=Path(sys.argv[1]); '
        'sys.path.insert(0,str(script.parent)); '
        'sys.argv=[str(script)]; '
        'runpy.run_path(str(script),run_name="__main__")'
    )
    return subprocess.call([sys.executable, '-B', '-c', runner, str(script)], cwd=ROOT)

if __name__=='__main__':sys.exit(main())
