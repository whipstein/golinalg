package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dtrt03 computes the residual for the solution to a scaled triangular
// system of equations A*x = s*b  or  A'*x = s*b.
// Here A is a triangular matrix, A' is the transpose of A, s is a
// scalar, and x and b are N by NRHS matrices.  The test ratio is the
// maximum over the number of right hand sides of
//    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A or A' and EPS is the machine epsilon.
func dtrt03(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, a *mat.Matrix, scale float64, cnorm *mat.Vector, tscal float64, x, b *mat.Matrix, work *mat.Vector) (resid float64) {
	var bignum, eps, err2, one, smlnum, tnorm, xnorm, xscal, zero float64
	var ix, j int
	var err error

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0
	if n <= 0 || nrhs <= 0 {
		resid = zero
		return
	}
	eps = golapack.Dlamch(Epsilon)
	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     Compute the norm of the triangular matrix A using the column
	//     norms already computed by DLATRS.
	tnorm = zero
	if diag == NonUnit {
		for j = 1; j <= n; j++ {
			tnorm = math.Max(tnorm, tscal*math.Abs(a.Get(j-1, j-1))+cnorm.Get(j-1))
		}
	} else {
		for j = 1; j <= n; j++ {
			tnorm = math.Max(tnorm, tscal+cnorm.Get(j-1))
		}
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ).
	resid = zero
	for j = 1; j <= nrhs; j++ {
		goblas.Dcopy(n, x.Vector(0, j-1, 1), work.Off(0, 1))
		ix = goblas.Idamax(n, work.Off(0, 1))
		xnorm = math.Max(one, math.Abs(x.Get(ix-1, j-1)))
		xscal = (one / xnorm) / float64(n)
		goblas.Dscal(n, xscal, work.Off(0, 1))
		if err = goblas.Dtrmv(uplo, trans, diag, n, a, work.Off(0, 1)); err != nil {
			panic(err)
		}
		goblas.Daxpy(n, -scale*xscal, b.Vector(0, j-1, 1), work.Off(0, 1))
		ix = goblas.Idamax(n, work.Off(0, 1))
		err2 = tscal * math.Abs(work.Get(ix-1))
		ix = goblas.Idamax(n, x.Vector(0, j-1, 1))
		xnorm = math.Abs(x.Get(ix-1, j-1))
		if err2*smlnum <= xnorm {
			if xnorm > zero {
				err2 /= xnorm
			}
		} else {
			if err2 > zero {
				err2 = one / eps
			}
		}
		if err2*smlnum <= tnorm {
			if tnorm > zero {
				err2 /= tnorm
			}
		} else {
			if err2 > zero {
				err2 = one / eps
			}
		}
		resid = math.Max(resid, err2)
	}

	return
}
