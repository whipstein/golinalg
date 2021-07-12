package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget07 tests the error bounds from iterative refinement for the
// computed solution to a system of equations op(A)*X = B, where A is a
// general n by n matrix and op(A) = A or A**T, depending on TRANS.
//
// RESLTS(1) = test of the error bound
//           = norm(X - XACT) / ( norm(X) * FERR )
//
// A large value is returned if this ratio is not less than one.
//
// RESLTS(2) = residual from the iterative refinement routine
//           = the maximum of BERR / ( (n+1)*EPS + (*) ), where
//             (*) = (n+1)*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
func Dget07(trans byte, n *int, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, xact *mat.Matrix, ldxact *int, ferr *mat.Vector, chkferr bool, berr, reslts *mat.Vector) {
	var notran bool
	var axbi, diff, eps, errbnd, one, ovfl, tmp, unfl, xnorm, zero float64
	var i, imax, j, k int

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
	notran = trans == 'N'

	//     Test 1:  Compute the maximum of
	//        norm(X - XACT) / ( norm(X) * FERR )
	//     over all the vectors X and XACT using the infinity-norm.
	errbnd = zero
	if chkferr {
		for j = 1; j <= (*nrhs); j++ {
			imax = goblas.Idamax(*n, x.Vector(0, j-1, 1))
			xnorm = math.Max(math.Abs(x.Get(imax-1, j-1)), unfl)
			diff = zero
			for i = 1; i <= (*n); i++ {
				diff = math.Max(diff, math.Abs(x.Get(i-1, j-1)-xact.Get(i-1, j-1)))
			}

			if xnorm > one {
				goto label20
			} else if diff <= ovfl*xnorm {
				goto label20
			} else {
				errbnd = one / eps
				goto label30
			}
			//
		label20:
			;
			if diff/xnorm <= ferr.Get(j-1) {
				errbnd = math.Max(errbnd, (diff/xnorm)/ferr.Get(j-1))
			} else {
				errbnd = one / eps
			}
		label30:
		}
	}
	reslts.Set(0, errbnd)

	//     Test 2:  Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
	//     (*) = (n+1)*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
	for k = 1; k <= (*nrhs); k++ {
		for i = 1; i <= (*n); i++ {
			tmp = math.Abs(b.Get(i-1, k-1))
			if notran {
				for j = 1; j <= (*n); j++ {
					tmp += math.Abs(a.Get(i-1, j-1)) * math.Abs(x.Get(j-1, k-1))
				}
			} else {
				for j = 1; j <= (*n); j++ {
					tmp += math.Abs(a.Get(j-1, i-1)) * math.Abs(x.Get(j-1, k-1))
				}
			}
			if i == 1 {
				axbi = tmp
			} else {
				axbi = math.Min(axbi, tmp)
			}
		}
		tmp = berr.Get(k-1) / (float64((*n)+1)*eps + float64((*n)+1)*unfl/math.Max(axbi, float64((*n)+1)*unfl))
		if k == 1 {
			reslts.Set(1, tmp)
		} else {
			reslts.Set(1, math.Max(reslts.Get(1), tmp))
		}
	}
}
