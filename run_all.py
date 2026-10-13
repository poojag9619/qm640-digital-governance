"""Run the whole data pipeline end to end.

    python run_all.py              # refresh PhonePe, load filled templates, rebuild panel
    python run_all.py --offline    # skip the PhonePe download (use the existing clone)
    python run_all.py --models     # also run the RQ1-RQ4 models (reports/model_results.md)
"""
import argparse

from src.clean import build_panel
from src.collect import manual_sources, phonepe_pulse
from src.features import build_features

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true")
ap.add_argument("--models", action="store_true")
a = ap.parse_args()

print("1/4 PhonePe Pulse")
if a.offline:
    q = phonepe_pulse.parse()
    q.to_csv(phonepe_pulse.INTERIM / "phonepe_state_quarter.csv", index=False)
    phonepe_pulse.to_year(q).to_csv(phonepe_pulse.INTERIM / "phonepe_state_year.csv", index=False)
else:
    phonepe_pulse.main()

print("\n2/4 Manual sources (templates + filled files)")
manual_sources.make_templates()
manual_sources.load_all()

print("\n3/4 Merge into state-year panel")
build_panel.main()

print("\n4/4 Derived variables and data dictionary")
build_features.main()
if a.models:
    from src.models import rq_models
    print("\n5/5 Models")
    rq_models.main()
print("\nDone. See reports/missing_data_report.md")
