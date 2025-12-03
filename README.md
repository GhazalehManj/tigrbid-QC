# Instruction to run streamlit app for Kimel 
1. Create python environment
```bash
cd /projects/username
module load python/3.10.7
python -m venv tigrbid_QC_env
source tigrbid_QC_env/bin/activate
pip install streamlit==1.49.1 pandas==2.3.2 pydantic==2.11.7

streamlit run /projects/ttan/SCanD_project/code/tigrbid-QC/app/main.py -- --fs_metric /projects/ttan/TAY/derivatives/freesurfer/7.4.1/00_group2_stats_tables/euler.tsv --fmri_dir /projects/ttan/TAY/derivatives/fmriprep/23.2.3/ --output_dir /projects/ttan/TAY_test_01/

```

