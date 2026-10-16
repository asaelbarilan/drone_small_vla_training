"""Render measured D257 smoke trajectories without changing the tested package."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'reports/d257_hf_release';flights=out/'flights'
turn=json.loads((flights/'sim_2025-03-27_17-34-24.json').read_text());ascent=json.loads((flights/'sim_2025-03-27_16-41-53.json').read_text());assert turn['complete'] and ascent['complete']
t=np.array(turn['poses']);a=np.array(ascent['poses']);ref=np.array(ascent['task']['reference_path_preprocessed']);fig,ax=plt.subplots(1,3,figsize=(15,4.6))
ax[0].plot(t[:,4],label='Downloaded VLA');ax[0].axhline(23.471,color='black',ls='--',label='Target heading');ax[0].set(xlabel='Executed move',ylabel='Heading change (degrees)',title='Turn: 23.471 degrees right')
ax[1].plot(a[:,0]/100,a[:,2]/100,label='Downloaded VLA');ax[1].plot(ref[:,0]/100,ref[:,2]/100,'k--',label='Target route');ax[1].set(xlabel='Forward (m)',ylabel='Altitude change (m)',title='Ascent: 9 m along 30-degree slope')
ax[2].plot(a[:,0]/100,a[:,1]/100,label='Downloaded VLA');ax[2].plot(ref[:,0]/100,ref[:,1]/100,'k--',label='Target route');ax[2].set(xlabel='Forward (m)',ylabel='Lateral (m)',title='Ascent: lateral deviation')
for axis in ax:axis.grid(alpha=.2);axis.legend()
fig.suptitle('Fresh private Hugging Face download — local NF4 smoke test (no training)');fig.tight_layout();fig.savefig(out/'smoke_paths.png',dpi=160);plt.close(fig)
print('Rendered smoke_paths.png')
