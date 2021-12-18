package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget10 compares two matrices A and B and computes the ratio
// RESULT = norm( A - B ) / ( norm(A) * M * EPS )
func dget10(m, n int, a, b *mat.Matrix, work *mat.Vector) float64 {
	var anorm, eps, one, result, unfl, wnorm, zero float64
	var j int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		result = zero
		return result
	}

	unfl = golapack.Dlamch(SafeMinimum)
	eps = golapack.Dlamch(Precision)

	wnorm = zero
	for j = 1; j <= n; j++ {
		work.Copy(m, a.Off(0, j-1).Vector(), 1, 1)
		work.Axpy(m, -one, b.Off(0, j-1).Vector(), 1, 1)
		wnorm = math.Max(wnorm, work.Asum(n, 1))
	}

	anorm = math.Max(golapack.Dlange('1', m, n, a, work), unfl)

	if anorm > wnorm {
		result = (wnorm / anorm) / (float64(m) * eps)
	} else {
		if anorm < one {
			result = (math.Min(wnorm, float64(m)*anorm) / anorm) / (float64(m) * eps)
		} else {
			result = math.Min(wnorm/anorm, float64(m)) / (float64(m) * eps)
		}
	}

	return result
}
