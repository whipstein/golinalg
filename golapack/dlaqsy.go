package golapack

import "github.com/whipstein/golinalg/mat"

// Dlaqsy equilibrates a symmetric matrix A using the scaling factors
// in the vector S.
func Dlaqsy(uplo byte, n *int, a *mat.Matrix, lda *int, s *mat.Vector, scond, amax *float64, equed *byte) {
	var cj, large, one, small, thresh float64
	var i, j int

	one = 1.0
	thresh = 0.1

	//     Quick return if possible
	if (*n) <= 0 {
		(*equed) = 'N'
		return
	}

	//     Initialize LARGE and SMALL.
	small = Dlamch(SafeMinimum) / Dlamch(Precision)
	large = one / small

	if (*scond) >= thresh && (*amax) >= small && (*amax) <= large {
		//        No equilibration
		(*equed) = 'N'
	} else {
		//        Replace A by diag(S) * A * diag(S).
		if uplo == 'U' {
			//           Upper triangle of A is stored.
			for j = 1; j <= (*n); j++ {
				cj = s.Get(j - 1)
				for i = 1; i <= j; i++ {
					a.Set(i-1, j-1, cj*s.Get(i-1)*a.Get(i-1, j-1))
				}
			}
		} else {
			//           Lower triangle of A is stored.
			for j = 1; j <= (*n); j++ {
				cj = s.Get(j - 1)
				for i = j; i <= (*n); i++ {
					a.Set(i-1, j-1, cj*s.Get(i-1)*a.Get(i-1, j-1))
				}
			}
		}
		(*equed) = 'Y'
	}
}
