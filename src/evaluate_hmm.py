import json, csv
from main import PARTITION, inventory, predict

if PARTITION!='student_test': raise SystemExit('Decoder experiments are restricted to student_test.')
rows=[]
for tid in sorted(inventory()):
    r=predict(tid); row={'trip_id':tid,**r['metrics']}; rows.append(row); print(row,flush=True)
with open('hmm_comparison.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for name in ('classifier_top1_accuracy','top1_accuracy','classifier_route_continuity','route_continuity'):
    values=[r[name] for r in rows if r[name] is not None]
    print(name, sum(values)/len(values) if values else 'N/A','(mean across trips)')
