package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget10 compares two matrices A and B and computes the ratio
// RESULT = norm( A - B ) / ( norm(A) * M * EPS )
func Dget10(m, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, work *mat.Vector, result *float64) {
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
		goblas.Dcopy(*m, a.Vector(0, j-1), 1, work, 1)
		goblas.Daxpy(*m, -one, b.Vector(0, j-1), 1, work, 1)
		wnorm = maxf64(wnorm, goblas.Dasum(*n, work, 1))
	}

	anorm = maxf64(golapack.Dlange('1', m, n, a, lda, work), unfl)

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
