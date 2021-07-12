package golapack

import "github.com/whipstein/golinalg/mat"

// Dlaqsb equilibrates a symmetric band matrix A using the scaling
// factors in the vector S.
func Dlaqsb(uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, s *mat.Vector, scond, amax *float64, equed *byte) {
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
			//           Upper triangle of A is stored in band format.
			for j = 1; j <= (*n); j++ {
				cj = s.Get(j - 1)
				for i = max(1, j-(*kd)); i <= j; i++ {
					ab.Set((*kd)+1+i-j-1, j-1, cj*s.Get(i-1)*ab.Get((*kd)+1+i-j-1, j-1))
				}
			}
		} else {
			//           Lower triangle of A is stored.
			for j = 1; j <= (*n); j++ {
				cj = s.Get(j - 1)
				for i = j; i <= min(*n, j+(*kd)); i++ {
					ab.Set(1+i-j-1, j-1, cj*s.Get(i-1)*ab.Get(1+i-j-1, j-1))
				}
			}
		}
		(*equed) = 'Y'
	}
}
