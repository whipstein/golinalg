package golapack

import "github.com/whipstein/golinalg/mat"

// Dlaqsp equilibrates a symmetric matrix A using the scaling factors
// in the vector S.
func Dlaqsp(uplo mat.MatUplo, n int, ap, s *mat.Vector, scond, amax float64) (equed byte) {
	var cj, large, one, small, thresh float64
	var i, j, jc int

	one = 1.0
	thresh = 0.1

	//     Quick return if possible
	if n <= 0 {
		equed = 'N'
		return
	}

	//     Initialize LARGE and SMALL.
	small = Dlamch(SafeMinimum) / Dlamch(Precision)
	large = one / small

	if scond >= thresh && amax >= small && amax <= large {
		//        No equilibration
		equed = 'N'
	} else {
		//        Replace A by diag(S) * A * diag(S).
		if uplo == Upper {
			//           Upper triangle of A is stored.
			jc = 1
			for j = 1; j <= n; j++ {
				cj = s.Get(j - 1)
				for i = 1; i <= j; i++ {
					ap.Set(jc+i-1-1, cj*s.Get(i-1)*ap.Get(jc+i-1-1))
				}
				jc = jc + j
			}
		} else {
			//           Lower triangle of A is stored.
			jc = 1
			for j = 1; j <= n; j++ {
				cj = s.Get(j - 1)
				for i = j; i <= n; i++ {
					ap.Set(jc+i-j-1, cj*s.Get(i-1)*ap.Get(jc+i-j-1))
				}
				jc = jc + n - j + 1
			}
		}
		equed = 'Y'
	}

	return
}
