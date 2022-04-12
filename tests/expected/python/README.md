# Generate expected data

Generated from the output of one run from the files in `tests/outputs/python`

```python
import numpy as np

tmp = np.load("TYPEB_FITHRF.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
np.save("TYPEB_FITHRF_HRFindex.npy", results)

tmp = np.load("TYPEC_FITHRF_GLMDENOISE.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
np.save("TYPEC_FITHRF_HRFindex.npy", results)

tmp = np.load("TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
np.save("TYPED_FITHRF_HRFindex.npy", results)

results = tmp.item(0)["R2"]
np.save("TYPED_FITHRF_R2.npy", results)
```

<!-- TODO ?
## For CSV

```python
import numpy as np

tmp = np.load("TYPEB_FITHRF.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
results = results.squeeze();
np.savetxt("TYPEB_FITHRF_HRFindex.csv", results, delimiter=",", fmt="%i")

tmp = np.load("TYPEC_FITHRF_GLMDENOISE.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
results = results.squeeze();
np.savetxt("TYPEC_FITHRF_HRFindex.csv", results, delimiter=",", fmt="%i")

tmp = np.load("TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True)
results = tmp.item(0)["HRFindex"]
results = results.squeeze();
np.savetxt("TYPED_FITHRF_HRFindex.csv", results, delimiter=",", fmt="%i")
```

Read from csv

```python
from numpy import genfromtxt

my_data = genfromtxt('TYPEB_FITHRF_HRFindex.csv', delimiter=',')
``` -->
