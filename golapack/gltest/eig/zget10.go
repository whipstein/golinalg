package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
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
		goblas.Zcopy(*m, a.CVector(0, j-1, 1), work.Off(0, 1))
		goblas.Zaxpy(*m, complex(-one, 0), b.CVector(0, j-1, 1), work.Off(0, 1))
		wnorm = math.Max(wnorm, goblas.Dzasum(*n, work.Off(0, 1)))
	}

	anorm = math.Max(golapack.Zlange('1', m, n, a, lda, rwork), unfl)

	if anorm > wnorm {
		(*result) = (wnorm / anorm) / (float64(*m) * eps)
	} else {
		if anorm < one {
			(*result) = (math.Min(wnorm, float64(*m)*anorm) / anorm) / (float64(*m) * eps)
		} else {
			(*result) = math.Min(wnorm/anorm, float64(*m)) / (float64(*m) * eps)
		}
	}
}
