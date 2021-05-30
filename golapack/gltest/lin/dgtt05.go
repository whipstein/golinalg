package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dgtt05 tests the error bounds from iterative refinement for the
// computed solution to a system of equations A*X = B, where A is a
// general tridiagonal matrix of order n and op(A) = A or A**T,
// depending on TRANS.
//
// RESLTS(1) = test of the error bound
//           = norm(X - XACT) / ( norm(X) * FERR )
//
// A large value is returned if this ratio is not less than one.
//
// RESLTS(2) = residual from the iterative refinement routine
//           = the maximum of BERR / ( NZ*EPS + (*) ), where
//             (*) = NZ*UNFL / (min_i (math.Abs(op(A))*math.Abs(X) +math.Abs(b))_i )
//             and NZ = max. number of nonzeros in any row of A, plus 1
func Dgtt05(trans byte, n, nrhs *int, dl, d, du *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, xact *mat.Matrix, ldxact *int, ferr, berr, reslts *mat.Vector) {
	var notran bool
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
	//
	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	notran = trans == 'N'
	nz = 4

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
		//
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

	//     Test 2:  Compute the maximum of BERR / ( NZ*EPS + (*) ), where
	//     (*) = NZ*UNFL / (min_i (math.Abs(op(A))*math.Abs(X) +math.Abs(b))_i )
	for k = 1; k <= (*nrhs); k++ {
		if notran {
			if (*n) == 1 {
				axbi = math.Abs(b.Get(0, k-1)) + math.Abs(d.Get(0)*x.Get(0, k-1))
			} else {
				axbi = math.Abs(b.Get(0, k-1)) + math.Abs(d.Get(0)*x.Get(0, k-1)) + math.Abs(du.Get(0)*x.Get(1, k-1))
				for i = 2; i <= (*n)-1; i++ {
					tmp = math.Abs(b.Get(i-1, k-1)) + math.Abs(dl.Get(i-1-1)*x.Get(i-1-1, k-1)) + math.Abs(d.Get(i-1)*x.Get(i-1, k-1)) + math.Abs(du.Get(i-1)*x.Get(i+1-1, k-1))
					axbi = minf64(axbi, tmp)
				}
				tmp = math.Abs(b.Get((*n)-1, k-1)) + math.Abs(dl.Get((*n)-1-1)*x.Get((*n)-1-1, k-1)) + math.Abs(d.Get((*n)-1)*x.Get((*n)-1, k-1))
				axbi = minf64(axbi, tmp)
			}
		} else {
			if (*n) == 1 {
				axbi = math.Abs(b.Get(0, k-1)) + math.Abs(d.Get(0)*x.Get(0, k-1))
			} else {
				axbi = math.Abs(b.Get(0, k-1)) + math.Abs(d.Get(0)*x.Get(0, k-1)) + math.Abs(dl.Get(0)*x.Get(1, k-1))
				for i = 2; i <= (*n)-1; i++ {
					tmp = math.Abs(b.Get(i-1, k-1)) + math.Abs(du.Get(i-1-1)*x.Get(i-1-1, k-1)) + math.Abs(d.Get(i-1)*x.Get(i-1, k-1)) + math.Abs(dl.Get(i-1)*x.Get(i+1-1, k-1))
					axbi = minf64(axbi, tmp)
				}
				tmp = math.Abs(b.Get((*n)-1, k-1)) + math.Abs(du.Get((*n)-1-1)*x.Get((*n)-1-1, k-1)) + math.Abs(d.Get((*n)-1)*x.Get((*n)-1, k-1))
				axbi = minf64(axbi, tmp)
			}
		}
		tmp = berr.Get(k-1) / (float64(nz)*eps + float64(nz)*unfl/maxf64(axbi, float64(nz)*unfl))
		if k == 1 {
			reslts.Set(1, tmp)
		} else {
			reslts.Set(1, maxf64(reslts.Get(1), tmp))
		}
	}
}
