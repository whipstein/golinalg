package golapack

import "github.com/whipstein/golinalg/mat"

// Zlaqgb equilibrates a general M by N band matrix A with KL
// subdiagonals and KU superdiagonals using the row and scaling factors
// in the vectors R and C.
func Zlaqgb(m, n, kl, ku *int, ab *mat.CMatrix, ldab *int, r, c *mat.Vector, rowcnd, colcnd, amax *float64, equed *byte) {
	var cj, large, one, small, thresh float64
	var i, j int

	one = 1.0
	thresh = 0.1

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		(*equed) = 'N'
		return
	}

	//     Initialize LARGE and SMALL.
	small = Dlamch(SafeMinimum) / Dlamch(Precision)
	large = one / small

	if (*rowcnd) >= thresh && (*amax) >= small && (*amax) <= large {
		//        No row scaling
		if (*colcnd) >= thresh {
			//           No column scaling
			(*equed) = 'N'
		} else {
			//           Column scaling
			for j = 1; j <= (*n); j++ {
				cj = c.Get(j - 1)
				for i = max(1, j-(*ku)); i <= min(*m, j+(*kl)); i++ {
					ab.Set((*ku)+1+i-j-1, j-1, complex(cj, 0)*ab.Get((*ku)+1+i-j-1, j-1))
				}
			}
			(*equed) = 'C'
		}
	} else if (*colcnd) >= thresh {
		//        Row scaling, no column scaling
		for j = 1; j <= (*n); j++ {
			for i = max(1, j-(*ku)); i <= min(*m, j+(*kl)); i++ {
				ab.Set((*ku)+1+i-j-1, j-1, r.GetCmplx(i-1)*ab.Get((*ku)+1+i-j-1, j-1))
			}
		}
		(*equed) = 'R'
	} else {
		//        Row and column scaling
		for j = 1; j <= (*n); j++ {
			cj = c.Get(j - 1)
			for i = max(1, j-(*ku)); i <= min(*m, j+(*kl)); i++ {
				ab.Set((*ku)+1+i-j-1, j-1, complex(cj*r.Get(i-1), 0)*ab.Get((*ku)+1+i-j-1, j-1))
			}
		}
		(*equed) = 'B'
	}
}
