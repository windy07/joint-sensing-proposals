"""Stage6 entry point. Large computation requires --execute."""
from pathlib import Path
import os,sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ACTIVE/MAINLINE_RESEARCH/spatial_scheduling'))
if __name__=='__main__':
    from stage6_server import main
    main()
