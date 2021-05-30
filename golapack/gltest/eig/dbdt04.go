package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dbdt04 reconstructs a bidiagonal matrix B from its (partial) SVD:
//    S = U' * B * V
// where U and V are orthogonal matrices and S is diagonal.
//
// The test ratio to test the singular value decomposition is
//    RESID = norm( S - U' * B * V ) / ( n * norm(B) * EPS )
// where VT = V' and EPS is the machine precision.
func Dbdt04(uplo byte, n *int, d, e, s *mat.Vector, ns *int, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, work *mat.Vector, resid *float64) {
	var bnorm, eps, one, zero float64
	var i, j, k int

	zero = 0.0
	one = 1.0

	//     Quick return if possible.
	(*resid) = zero
	if (*n) <= 0 || (*ns) <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)

	//     Compute S - U' * B * V.
	bnorm = zero

	if uplo == 'U' {
		//        B is upper bidiagonal.
		k = 0
		for i = 1; i <= (*ns); i++ {
			for j = 1; j <= (*n)-1; j++ {
				k = k + 1
				work.Set(k-1, d.Get(j-1)*vt.Get(i-1, j-1)+e.Get(j-1)*vt.Get(i-1, j+1-1))
			}
			k = k + 1
			work.Set(k-1, d.Get((*n)-1)*vt.Get(i-1, (*n)-1))
		}
		bnorm = math.Abs(d.Get(0))
		for i = 2; i <= (*n); i++ {
			bnorm = maxf64(bnorm, math.Abs(d.Get(i-1))+math.Abs(e.Get(i-1-1)))
		}
	} else {
		//        B is lower bidiagonal.
		k = 0
		for i = 1; i <= (*ns); i++ {
			k = k + 1
			work.Set(k-1, d.Get(0)*vt.Get(i-1, 0))
			for j = 1; j <= (*n)-1; j++ {
				k = k + 1
				work.Set(k-1, e.Get(j-1)*vt.Get(i-1, j-1)+d.Get(j+1-1)*vt.Get(i-1, j+1-1))
			}
		}
		bnorm = math.Abs(d.Get((*n) - 1))
		for i = 1; i <= (*n)-1; i++ {
			bnorm = maxf64(bnorm, math.Abs(d.Get(i-1))+math.Abs(e.Get(i-1)))
		}
	}

	goblas.Dgemm(Trans, NoTrans, ns, ns, n, toPtrf64(-one), u, ldu, work.Matrix(*n, opts), n, &zero, work.MatrixOff(1+(*n)*(*ns)-1, *ns, opts), ns)

	//     norm(S - U' * B * V)
	k = (*n) * (*ns)
	for i = 1; i <= (*ns); i++ {
		work.Set(k+i-1, work.Get(k+i-1)+s.Get(i-1))
		(*resid) = maxf64(*resid, goblas.Dasum(ns, work.Off(k+1-1), toPtr(1)))
		k = k + (*ns)
	}

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
