package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dbdt03 reconstructs a bidiagonal matrix B from its SVD:
//    S = U' * B * V
// where U and V are orthogonal matrices and S is diagonal.
//
// The test ratio to test the singular value decomposition is
//    RESID = norm( B - U * S * VT ) / ( n * norm(B) * EPS )
// where VT = V' and EPS is the machine precision.
func Dbdt03(uplo byte, n, kd *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, s *mat.Vector, vt *mat.Matrix, ldvt *int, work *mat.Vector, resid *float64) {
	var bnorm, eps, one, zero float64
	var i, j int

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	(*resid) = zero
	if (*n) <= 0 {
		return
	}

	//     Compute B - U * S * V' one column at a time.
	bnorm = zero
	if (*kd) >= 1 {
		//        B is bidiagonal.
		if uplo == 'U' {
			//           B is upper bidiagonal.
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= (*n); i++ {
					work.Set((*n)+i-1, s.Get(i-1)*vt.Get(i-1, j-1))
				}
				goblas.Dgemv(NoTrans, n, n, toPtrf64(-one), u, ldu, work.Off((*n)+1-1), toPtr(1), &zero, work, toPtr(1))
				work.Set(j-1, work.Get(j-1)+d.Get(j-1))
				if j > 1 {
					work.Set(j-1-1, work.Get(j-1-1)+e.Get(j-1-1))
					bnorm = maxf64(bnorm, math.Abs(d.Get(j-1))+math.Abs(e.Get(j-1-1)))
				} else {
					bnorm = maxf64(bnorm, math.Abs(d.Get(j-1)))
				}
				(*resid) = maxf64(*resid, goblas.Dasum(n, work, toPtr(1)))
			}
		} else {
			//           B is lower bidiagonal.
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= (*n); i++ {
					work.Set((*n)+i-1, s.Get(i-1)*vt.Get(i-1, j-1))
				}
				goblas.Dgemv(NoTrans, n, n, toPtrf64(-one), u, ldu, work.Off((*n)+1-1), toPtr(1), &zero, work, toPtr(1))
				work.Set(j-1, work.Get(j-1)+d.Get(j-1))
				if j < (*n) {
					work.Set(j+1-1, work.Get(j+1-1)+e.Get(j-1))
					bnorm = maxf64(bnorm, math.Abs(d.Get(j-1))+math.Abs(e.Get(j-1)))
				} else {
					bnorm = maxf64(bnorm, math.Abs(d.Get(j-1)))
				}
				(*resid) = maxf64(*resid, goblas.Dasum(n, work, toPtr(1)))
			}
		}
	} else {
		//        B is diagonal.
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*n); i++ {
				work.Set((*n)+i-1, s.Get(i-1)*vt.Get(i-1, j-1))
			}
			goblas.Dgemv(NoTrans, n, n, toPtrf64(-one), u, ldu, work.Off((*n)+1-1), toPtr(1), &zero, work, toPtr(1))
			work.Set(j-1, work.Get(j-1)+d.Get(j-1))
			(*resid) = maxf64(*resid, goblas.Dasum(n, work, toPtr(1)))
		}
		j = goblas.Idamax(n, d, toPtr(1))
		bnorm = math.Abs(d.Get(j - 1))
	}

	//     Compute norm(B - U * S * V') / ( n * norm(B) * EPS )
	eps = golapack.Dlamch(Precision)

	if bnorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		if bnorm >= (*resid) {
			(*resid) = ((*resid) / bnorm) / (float64(*n) * eps)
		} else {
			if bnorm < one {
				(*resid) = (minf64(*resid, float64(*n)*bnorm) / bnorm) / (float64(*n) * eps)
			} else {
				(*resid) = minf64((*resid)/bnorm, float64(*n)) / (float64(*n) * eps)
			}
		}
	}
}
