package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dpbt05 tests the error bounds from iterative refinement for the
// computed solution to a system of equations A*X = B, where A is a
// symmetric band matrix.
//
// RESLTS(1) = test of the error bound
//           = norm(X - XACT) / ( norm(X) * FERR )
//
// A large value is returned if this ratio is not less than one.
//
// RESLTS(2) = residual from the iterative refinement routine
//           = the maximum of BERR / ( NZ*EPS + (*) ), where
//             (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
//             and NZ = math.Max. number of nonzeros in any row of A, plus 1
func Dpbt05(uplo byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, xact *mat.Matrix, ldxact *int, ferr, berr, reslts *mat.Vector) {
	var upper bool
	var axbi, diff, eps, errbnd, one, ovfl, tmp, unfl, xnorm, zero float64
	var i, imax, j, k, nz int

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
	nz = 2*max(*kd, (*n)-1) + 1

	//     Test 1:  Compute the maximum of
	//        norm(X - XACT) / ( norm(X) * FERR )
	//     over all the vectors X and XACT using the infinity-norm.
	errbnd = zero
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

	label20:
		;
		if diff/xnorm <= ferr.Get(j-1) {
			errbnd = math.Max(errbnd, (diff/xnorm)/ferr.Get(j-1))
		} else {
			errbnd = one / eps
		}
	label30:
	}
	reslts.Set(0, errbnd)

	//     Test 2:  Compute the maximum of BERR / ( NZ*EPS + (*) ), where
	//     (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
	for k = 1; k <= (*nrhs); k++ {
		for i = 1; i <= (*n); i++ {
			tmp = math.Abs(b.Get(i-1, k-1))
			if upper {
				for j = max(i-(*kd), 1); j <= i; j++ {
					tmp = tmp + math.Abs(ab.Get((*kd)+1-i+j-1, i-1))*math.Abs(x.Get(j-1, k-1))
				}
				for j = i + 1; j <= min(i+(*kd), *n); j++ {
					tmp = tmp + math.Abs(ab.Get((*kd)+1+i-j-1, j-1))*math.Abs(x.Get(j-1, k-1))
				}
			} else {
				for j = max(i-(*kd), 1); j <= i-1; j++ {
					tmp = tmp + math.Abs(ab.Get(1+i-j-1, j-1))*math.Abs(x.Get(j-1, k-1))
				}
				for j = i; j <= min(i+(*kd), *n); j++ {
					tmp = tmp + math.Abs(ab.Get(1+j-i-1, i-1))*math.Abs(x.Get(j-1, k-1))
				}
			}
			if i == 1 {
				axbi = tmp
			} else {
				axbi = math.Min(axbi, tmp)
			}
		}
		tmp = berr.Get(k-1) / (float64(nz)*eps + float64(nz)*unfl/math.Max(axbi, float64(nz)*unfl))
		if k == 1 {
			reslts.Set(1, tmp)
		} else {
			reslts.Set(1, math.Max(reslts.Get(1), tmp))
		}
	}
}
