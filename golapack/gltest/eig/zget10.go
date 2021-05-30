package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zget10 compares two matrices A and B and computes the ratio
// RESULT = norm( A - B ) / ( norm(A) * M * EPS )
func Zget10(m, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, work *mat.CVector, rwork *mat.Vector, result *float64) {
	var anorm, eps, one, unfl, wnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		(*result) = zero
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	eps = golapack.Dlamch(Precision)

	wnorm = zero
	for j = 1; j <= (*n); j++ {
		goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		goblas.Zaxpy(m, toPtrc128(complex(-one, 0)), b.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		wnorm = maxf64(wnorm, goblas.Dzasum(n, work, func() *int { y := 1; return &y }()))
	}

	anorm = maxf64(golapack.Zlange('1', m, n, a, lda, rwork), unfl)

	if anorm > wnorm {
		(*result) = (wnorm / anorm) / (float64(*m) * eps)
	} else {
		if anorm < one {
			(*result) = (minf64(wnorm, float64(*m)*anorm) / anorm) / (float64(*m) * eps)
		} else {
			(*result) = minf64(wnorm/anorm, float64(*m)) / (float64(*m) * eps)
		}
	}
}
