package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dtrt05 tests the error bounds from iterative refinement for the
// computed solution to a system of equations A*X = B, where A is a
// triangular n by n matrix.
//
// RESLTS(1) = test of the error bound
//           = norm(X - XACT) / ( norm(X) * FERR )
//
// A large value is returned if this ratio is not less than one.
//
// RESLTS(2) = residual from the iterative refinement routine
//           = the maximum of BERR / ( (n+1)*EPS + (*) ), where
//             (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
func Dtrt05(uplo, trans, diag byte, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, xact *mat.Matrix, ldxact *int, ferr, berr, reslts *mat.Vector) {
	var notran, unit, upper bool
	var axbi, diff, eps, errbnd, one, ovfl, tmp, unfl, xnorm, zero float64
	var i, ifu, imax, j, k int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0.
	if (*n) <= 0 || (*nrhs) <= 0 {
		reslts.Set(0, zero)
		reslts.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	upper = uplo == 'U'
	notran = trans == 'N'
	unit = diag == 'U'

	//     Test 1:  Compute the maximum of
	//        norm(X - XACT) / ( norm(X) * FERR )
	//     over all the vectors X and XACT using the infinity-norm.
	errbnd = zero
	for j = 1; j <= (*nrhs); j++ {
		imax = goblas.Idamax(n, x.Vector(0, j-1), toPtr(1))
		xnorm = maxf64(math.Abs(x.Get(imax-1, j-1)), unfl)
		diff = zero
		for i = 1; i <= (*n); i++ {
			diff = maxf64(diff, math.Abs(x.Get(i-1, j-1)-xact.Get(i-1, j-1)))
		}

		if xnorm > one {
			goto label20
		} else if diff <= ovfl*xnorm {
			goto label20
		} else {
			errbnd = one / eps
			goto label30
		}

	label20:
		;
		if diff/xnorm <= ferr.Get(j-1) {
			errbnd = maxf64(errbnd, (diff/xnorm)/ferr.Get(j-1))
		} else {
			errbnd = one / eps
		}
	label30:
	}
	reslts.Set(0, errbnd)

	//     Test 2:  Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
	//     (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
	ifu = 0
	if unit {
		ifu = 1
	}
	for k = 1; k <= (*nrhs); k++ {
		for i = 1; i <= (*n); i++ {
			tmp = math.Abs(b.Get(i-1, k-1))
			if upper {
				if !notran {
					for j = 1; j <= i-ifu; j++ {
						tmp = tmp + math.Abs(a.Get(j-1, i-1))*math.Abs(x.Get(j-1, k-1))
					}
					if unit {
						tmp = tmp + math.Abs(x.Get(i-1, k-1))
					}
				} else {
					if unit {
						tmp = tmp + math.Abs(x.Get(i-1, k-1))
					}
					for j = i + ifu; j <= (*n); j++ {
						tmp = tmp + math.Abs(a.Get(i-1, j-1))*math.Abs(x.Get(j-1, k-1))
					}
				}
			} else {
				if notran {
					for j = 1; j <= i-ifu; j++ {
						tmp = tmp + math.Abs(a.Get(i-1, j-1))*math.Abs(x.Get(j-1, k-1))
					}
					if unit {
						tmp = tmp + math.Abs(x.Get(i-1, k-1))
					}
				} else {
					if unit {
						tmp = tmp + math.Abs(x.Get(i-1, k-1))
					}
					for j = i + ifu; j <= (*n); j++ {
						tmp = tmp + math.Abs(a.Get(j-1, i-1))*math.Abs(x.Get(j-1, k-1))
					}
				}
			}
			if i == 1 {
				axbi = tmp
			} else {
				axbi = minf64(axbi, tmp)
			}
		}
		tmp = berr.Get(k-1) / (float64((*n)+1)*eps + float64((*n)+1)*unfl/maxf64(axbi, float64((*n)+1)*unfl))
		if k == 1 {
			reslts.Set(1, tmp)
		} else {
			reslts.Set(1, maxf64(reslts.Get(1), tmp))
		}
	}
}
