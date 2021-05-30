package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Zbdt01 reconstructs a general matrix A from its bidiagonal form
//    A = Q * B * P'
// where Q (m by min(m,n)) and P' (min(m,n) by n) are unitary
// matrices and B is bidiagonal.
//
// The test ratio to test the reduction is
//    RESID = norm( A - Q * B * PT ) / ( n * norm(A) * EPS )
// where PT = P' and EPS is the machine precision.
func Zbdt01(m, n, kd *int, a *mat.CMatrix, lda *int, q *mat.CMatrix, ldq *int, d, e *mat.Vector, pt *mat.CMatrix, ldpt *int, work *mat.CVector, rwork *mat.Vector, resid *float64) {
	var anorm, eps, one, zero float64
	var i, j int

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		(*resid) = zero
		return
	}

	//     Compute A - Q * B * P' one column at a time.
	(*resid) = zero
	if (*kd) != 0 {
		//        B is bidiagonal.
		if (*kd) != 0 && (*m) >= (*n) {
			//           B is upper bidiagonal and M >= N.
			for j = 1; j <= (*n); j++ {
				goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*n)-1; i++ {
					work.Set((*m)+i-1, d.GetCmplx(i-1)*pt.Get(i-1, j-1)+e.GetCmplx(i-1)*pt.Get(i+1-1, j-1))
				}
				work.Set((*m)+(*n)-1, d.GetCmplx((*n)-1)*pt.Get((*n)-1, j-1))
				goblas.Zgemv(NoTrans, m, n, toPtrc128(-toCmplx(one)), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), toPtrc128(toCmplx(one)), work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else if (*kd) < 0 {
			//           B is upper bidiagonal and M < N.
			for j = 1; j <= (*n); j++ {
				goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*m)-1; i++ {
					work.Set((*m)+i-1, d.GetCmplx(i-1)*pt.Get(i-1, j-1)+e.GetCmplx(i-1)*pt.Get(i+1-1, j-1))
				}
				work.Set((*m)+(*m)-1, d.GetCmplx((*m)-1)*pt.Get((*m)-1, j-1))
				goblas.Zgemv(NoTrans, m, m, toPtrc128(-toCmplx(one)), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), toPtrc128(toCmplx(one)), work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else {
			//           B is lower bidiagonal.
			for j = 1; j <= (*n); j++ {
				goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				work.Set((*m)+1-1, d.GetCmplx(0)*pt.Get(0, j-1))
				for i = 2; i <= (*m); i++ {
					work.Set((*m)+i-1, e.GetCmplx(i-1-1)*pt.Get(i-1-1, j-1)+d.GetCmplx(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Zgemv(NoTrans, m, m, toPtrc128(-toCmplx(one)), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), toPtrc128(toCmplx(one)), work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
			}
		}
	} else {
		//        B is diagonal.
		if (*m) >= (*n) {
			for j = 1; j <= (*n); j++ {
				goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*n); i++ {
					work.Set((*m)+i-1, d.GetCmplx(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Zgemv(NoTrans, m, n, toPtrc128(-toCmplx(one)), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), toPtrc128(toCmplx(one)), work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				goblas.Zcopy(m, a.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*m); i++ {
					work.Set((*m)+i-1, d.GetCmplx(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Zgemv(NoTrans, m, m, toPtrc128(-toCmplx(one)), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), toPtrc128(toCmplx(one)), work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dzasum(m, work, func() *int { y := 1; return &y }()))
			}
		}
	}

	//     Compute norm(A - Q * B * P') / ( n * norm(A) * EPS )
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	eps = golapack.Dlamch(Precision)

	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		if anorm >= (*resid) {
			(*resid) = ((*resid) / anorm) / (float64(*n) * eps)
		} else {
			if anorm < one {
				(*resid) = (minf64(*resid, float64(*n)*anorm) / anorm) / (float64(*n) * eps)
			} else {
				(*resid) = minf64((*resid)/anorm, float64(*n)) / (float64(*n) * eps)
			}
		}
	}
}
