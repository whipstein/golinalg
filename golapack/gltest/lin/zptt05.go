package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zptt05 tests the error bounds from iterative refinement for the
// computed solution to a system of equations A*X = B, where A is a
// Hermitian tridiagonal matrix of order n.
//
// RESLTS(1) = test of the error bound
//           = norm(X - XACT) / ( norm(X) * FERR )
//
// A large value is returned if this ratio is not less than one.
//
// RESLTS(2) = residual from the iterative refinement routine
//           = the maximum of BERR / ( NZ*EPS + (*) ), where
//             (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
//             and NZ = max. number of nonzeros in any row of A, plus 1
func zptt05(n, nrhs int, d *mat.Vector, e *mat.CVector, b, x, xact *mat.CMatrix, ferr, berr, reslts *mat.Vector) {
	var axbi, diff, eps, errbnd, one, ovfl, tmp, unfl, xnorm, zero float64
	var i, imax, j, k, nz int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0 or NRHS = 0.
	if n <= 0 || nrhs <= 0 {
		reslts.Set(0, zero)
		reslts.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	nz = 4

	//     Test 1:  Compute the maximum of
	//        norm(X - XACT) / ( norm(X) * FERR )
	//     over all the vectors X and XACT using the infinity-norm.
	errbnd = zero
	for j = 1; j <= nrhs; j++ {
		imax = goblas.Izamax(n, x.CVector(0, j-1, 1))
		xnorm = math.Max(cabs1(x.Get(imax-1, j-1)), unfl)
		diff = zero
		for i = 1; i <= n; i++ {
			diff = math.Max(diff, cabs1(x.Get(i-1, j-1)-xact.Get(i-1, j-1)))
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
	for k = 1; k <= nrhs; k++ {
		if n == 1 {
			axbi = cabs1(b.Get(0, k-1)) + cabs1(d.GetCmplx(0)*x.Get(0, k-1))
		} else {
			axbi = cabs1(b.Get(0, k-1)) + cabs1(d.GetCmplx(0)*x.Get(0, k-1)) + cabs1(e.Get(0))*cabs1(x.Get(1, k-1))
			for i = 2; i <= n-1; i++ {
				tmp = cabs1(b.Get(i-1, k-1)) + cabs1(e.Get(i-1-1))*cabs1(x.Get(i-1-1, k-1)) + cabs1(d.GetCmplx(i-1)*x.Get(i-1, k-1)) + cabs1(e.Get(i-1))*cabs1(x.Get(i, k-1))
				axbi = math.Min(axbi, tmp)
			}
			tmp = cabs1(b.Get(n-1, k-1)) + cabs1(e.Get(n-1-1))*cabs1(x.Get(n-1-1, k-1)) + cabs1(d.GetCmplx(n-1)*x.Get(n-1, k-1))
			axbi = math.Min(axbi, tmp)
		}
		tmp = berr.Get(k-1) / (float64(nz)*eps + float64(nz)*unfl/math.Max(axbi, float64(nz)*unfl))
		if k == 1 {
			reslts.Set(1, tmp)
		} else {
			reslts.Set(1, math.Max(reslts.Get(1), tmp))
		}
	}
}
