import numpy as np
import json, codecs

a = np.array([1,1,1,2,3])
d = {}

### save np array in json
b = a.tolist() # nested lists with same data, indices
d['1'] = b

with open("my_1.json", "w") as fh:
        json.dump(d, fh, indent=4)

