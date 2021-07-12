package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Ztbt03 computes the residual for the solution to a scaled triangular
// system of equations  A*x = s*b,  A**T *x = s*b,  or  A**H *x = s*b
// when A is a triangular band matrix.  Here A**T  denotes the transpose
// of A, A**H denotes the conjugate transpose of A, s is a scalar, and
// x and b are N by NRHS matrices.  The test ratio is the maximum over
// the number of right hand sides of
//    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A, A**T, or A**H, and EPS is the machine epsilon.
func Ztbt03(uplo, trans, diag byte, n, kd, nrhs *int, ab *mat.CMatrix, ldab *int, scale *float64, cnorm *mat.Vector, tscal *float64, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, work *mat.CVector, resid *float64) {
	var eps, err2, one, smlnum, tnorm, xnorm, xscal, zero float64
	var ix, j int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}
	eps = golapack.Dlamch(Epsilon)
	smlnum = golapack.Dlamch(SafeMinimum)

	//     Compute the norm of the triangular matrix A using the column
	//     norms already computed by ZLATBS.
	tnorm = zero
	if diag == 'N' {
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				tnorm = math.Max(tnorm, (*tscal)*ab.GetMag((*kd), j-1)+cnorm.Get(j-1))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				tnorm = math.Max(tnorm, (*tscal)*ab.GetMag(0, j-1)+cnorm.Get(j-1))
			}
		}
	} else {
		for j = 1; j <= (*n); j++ {
			tnorm = math.Max(tnorm, (*tscal)+cnorm.Get(j-1))
		}
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ).
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		goblas.Zcopy(*n, x.CVector(0, j-1, 1), work.Off(0, 1))
		ix = goblas.Izamax(*n, work.Off(0, 1))
		xnorm = math.Max(one, x.GetMag(ix-1, j-1))
		xscal = (one / xnorm) / float64((*kd)+1)
		goblas.Zdscal(*n, xscal, work.Off(0, 1))
		err = goblas.Ztbmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, *kd, ab, work.Off(0, 1))
		goblas.Zaxpy(*n, complex(-(*scale)*xscal, 0), b.CVector(0, j-1, 1), work.Off(0, 1))
		ix = goblas.Izamax(*n, work.Off(0, 1))
		err2 = (*tscal) * work.GetMag(ix-1)
		ix = goblas.Izamax(*n, x.CVector(0, j-1, 1))
		xnorm = x.GetMag(ix-1, j-1)
		if err2*smlnum <= xnorm {
			if xnorm > zero {
				err2 = err2 / xnorm
			}
		} else {
			if err2 > zero {
				err2 = one / eps
			}
		}
		if err2*smlnum <= tnorm {
			if tnorm > zero {
				err2 = err2 / tnorm
			}
		} else {
			if err2 > zero {
				err2 = one / eps
			}
		}
		(*resid) = math.Max(*resid, err2)
	}
}
