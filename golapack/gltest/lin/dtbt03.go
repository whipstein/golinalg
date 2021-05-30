package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dtbt03 computes the residual for the solution to a scaled triangular
// system of equations  A*x = s*b  or  A'*x = s*b  when A is a
// triangular band matrix. Here A' is the transpose of A, s is a scalar,
// and x and b are N by NRHS matrices.  The test ratio is the maximum
// over the number of right hand sides of
//    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A or A' and EPS is the machine epsilon.
func Dtbt03(uplo, trans, diag byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, scale *float64, cnorm *mat.Vector, tscal *float64, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, work *mat.Vector, resid *float64) {
	var bignum, eps, err, one, smlnum, tnorm, xnorm, xscal, zero float64
	var ix, j int

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}
	eps = golapack.Dlamch(Epsilon)
	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Compute the norm of the triangular matrix A using the column
	//     norms already computed by DLATBS.
	tnorm = zero
	if diag == 'N' {
		if uplo == 'U' {
			for j = 1; j <= (*n); j++ {
				tnorm = maxf64(tnorm, (*tscal)*math.Abs(ab.Get((*kd)+1-1, j-1))+cnorm.Get(j-1))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				tnorm = maxf64(tnorm, (*tscal)*math.Abs(ab.Get(0, j-1))+cnorm.Get(j-1))
			}
		}
	} else {
		for j = 1; j <= (*n); j++ {
			tnorm = maxf64(tnorm, (*tscal)+cnorm.Get(j-1))
		}
	}

	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ).
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		goblas.Dcopy(n, x.Vector(0, j-1), toPtr(1), work, toPtr(1))
		ix = goblas.Idamax(n, work, toPtr(1))
		xnorm = maxf64(one, math.Abs(x.Get(ix-1, j-1)))
		xscal = (one / xnorm) / float64((*kd)+1)
		goblas.Dscal(n, &xscal, work, toPtr(1))
		goblas.Dtbmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, kd, ab, ldab, work, toPtr(1))
		goblas.Daxpy(n, toPtrf64(-(*scale)*xscal), b.Vector(0, j-1), toPtr(1), work, toPtr(1))
		ix = goblas.Idamax(n, work, toPtr(1))
		err = (*tscal) * math.Abs(work.Get(ix-1))
		ix = goblas.Idamax(n, x.Vector(0, j-1), toPtr(1))
		xnorm = math.Abs(x.Get(ix-1, j-1))
		if err*smlnum <= xnorm {
			if xnorm > zero {
				err = err / xnorm
			}
		} else {
			if err > zero {
				err = one / eps
			}
		}
		if err*smlnum <= tnorm {
			if tnorm > zero {
				err = err / tnorm
			}
		} else {
			if err > zero {
				err = one / eps
			}
		}
		(*resid) = maxf64(*resid, err)
	}
}
