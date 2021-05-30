package golapack

import "golinalg/mat"

// Dlaqge equilibrates a general M by N matrix A using the row and
// column scaling factors in the vectors R and C.
func Dlaqge(m, n *int, a *mat.Matrix, lda *int, r, c *mat.Vector, rowcnd *float64, colcnd *float64, amax *float64, equed *byte) {
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
				for i = 1; i <= (*m); i++ {
					a.Set(i-1, j-1, cj*a.Get(i-1, j-1))
				}
			}
			(*equed) = 'C'
		}
	} else if (*colcnd) >= thresh {
		//        Row scaling, no column scaling
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, r.Get(i-1)*a.Get(i-1, j-1))
			}
		}
		(*equed) = 'R'
	} else {
		//        Row and column scaling
		for j = 1; j <= (*n); j++ {
			cj = c.Get(j - 1)
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, cj*r.Get(i-1)*a.Get(i-1, j-1))
			}
		}
		(*equed) = 'B'
	}
}
