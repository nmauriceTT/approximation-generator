# Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

The following command will compute multiple approximation of several functions (`exp`, `log1p`, `tanh`) on given ranges (e.g. `[0, 1]` for `exp`) and plots the results (values and ULP error).

```
python approx.py
```

On top of this, a short report with polynomial expression will be generated along with the plots. 

Approximation methods include Chebyshev polynomials (np.chebyshev.chebyfit) as well as minmax approximation (scipy.optimize.minimize).



# TODOs

- [ ] Parallelize approximation generation and plotting
- [ ] Run exhaustive tests and plots
- [ ] Be more consistent with data type
- [ ] Improve formatting of function serialization