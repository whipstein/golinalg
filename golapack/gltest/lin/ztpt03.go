package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Ztpt03 computes the residual for the solution to a scaled triangular
// system of equations A*x = s*b,  A**T *x = s*b,  or  A**H *x = s*b,
// when the triangular matrix A is stored in packed format.  Here A**T
// denotes the transpose of A, A**H denotes the conjugate transpose of
// A, s is a scalar, and x and b are N by NRHS matrices.  The test ratio
// is the maximum over the number of right hand sides of
//    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
// where op(A) denotes A, A**T, or A**H, and EPS is the machine epsilon.
func Ztpt03(uplo, trans, diag byte, n, nrhs *int, ap *mat.CVector, scale *float64, cnorm *mat.Vector, tscal *float64, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, work *mat.CVector, resid *float64) {
	var eps, err, one, smlnum, tnorm, xnorm, xscal, zero float64
	var ix, j, jj int

	one = 1.0
	zero = 0.0

	//     Quick exit if N = 0.
	if (*n) <= 0 || (*nrhs) <= 0 {
		(*resid) = zero
		return
	}
	eps = golapack.Dlamch(Epsilon)
	smlnum = golapack.Dlamch(SafeMinimum)

	//     Compute the norm of the triangular matrix A using the column
	//     norms already computed by ZLATPS.
	tnorm = 0.
	if diag == 'N' {
		if uplo == 'U' {
			jj = 1
			for j = 1; j <= (*n); j++ {
				tnorm = maxf64(tnorm, (*tscal)*ap.GetMag(jj-1)+cnorm.Get(j-1))
				jj = jj + j
			}
		} else {
			jj = 1
			for j = 1; j <= (*n); j++ {
				tnorm = maxf64(tnorm, (*tscal)*ap.GetMag(jj-1)+cnorm.Get(j-1))
				jj = jj + (*n) - j + 1
			}
		}
	} else {
		for j = 1; j <= (*n); j++ {
			tnorm = maxf64(tnorm, (*tscal)+cnorm.Get(j-1))
		}
	}
	//
	//     Compute the maximum over the number of right hand sides of
	//        norm(op(A)*x - s*b) / ( norm(A) * norm(x) * EPS ).
	//
	(*resid) = zero
	for j = 1; j <= (*nrhs); j++ {
		goblas.Zcopy(n, x.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		ix = goblas.Izamax(n, work, func() *int { y := 1; return &y }())
		xnorm = maxf64(one, x.GetMag(ix-1, j-1))
		xscal = (one / xnorm) / float64(*n)
		goblas.Zdscal(n, &xscal, work, func() *int { y := 1; return &y }())
		goblas.Ztpmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, ap, work, func() *int { y := 1; return &y }())
		goblas.Zaxpy(n, toPtrc128(complex(-(*scale)*xscal, 0)), b.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		ix = goblas.Izamax(n, work, func() *int { y := 1; return &y }())
		err = (*tscal) * work.GetMag(ix-1)
		ix = goblas.Izamax(n, x.CVector(0, j-1), func() *int { y := 1; return &y }())
		xnorm = x.GetMag(ix-1, j-1)
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
