package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dbdt01 reconstructs a general matrix A from its bidiagonal form
//    A = Q * B * P'
// where Q (m by minf64(m,n)) and P' (minf64(m,n) by n) are orthogonal
// matrices and B is bidiagonal.
//
// The test ratio to test the reduction is
//    RESID = norm( A - Q * B * PT ) / ( n * norm(A) * EPS )
// where PT = P' and EPS is the machine precision.
func Dbdt01(m, n, kd *int, a *mat.Matrix, lda *int, q *mat.Matrix, ldq *int, d, e *mat.Vector, pt *mat.Matrix, ldpt *int, work *mat.Vector, resid *float64) {
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
				goblas.Dcopy(m, a.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*n)-1; i++ {
					work.Set((*m)+i-1, d.Get(i-1)*pt.Get(i-1, j-1)+e.Get(i-1)*pt.Get(i+1-1, j-1))
				}
				work.Set((*m)+(*n)-1, d.Get((*n)-1)*pt.Get((*n)-1, j-1))
				goblas.Dgemv(NoTrans, m, n, toPtrf64(-one), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), &one, work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else if (*kd) < 0 {
			//           B is upper bidiagonal and M < N.
			for j = 1; j <= (*n); j++ {
				goblas.Dcopy(m, a.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*m)-1; i++ {
					work.Set((*m)+i-1, d.Get(i-1)*pt.Get(i-1, j-1)+e.Get(i-1)*pt.Get(i+1-1, j-1))
				}
				work.Set((*m)+(*m)-1, d.Get((*m)-1)*pt.Get((*m)-1, j-1))
				goblas.Dgemv(NoTrans, m, m, toPtrf64(-one), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), &one, work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else {
			//           B is lower bidiagonal.
			for j = 1; j <= (*n); j++ {
				goblas.Dcopy(m, a.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				work.Set((*m)+1-1, d.Get(0)*pt.Get(0, j-1))
				for i = 2; i <= (*m); i++ {
					work.Set((*m)+i-1, e.Get(i-1-1)*pt.Get(i-1-1, j-1)+d.Get(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Dgemv(NoTrans, m, m, toPtrf64(-one), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), &one, work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dasum(m, work, func() *int { y := 1; return &y }()))
			}
		}
	} else {
		//        B is diagonal.
		if (*m) >= (*n) {
			for j = 1; j <= (*n); j++ {
				goblas.Dcopy(m, a.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*n); i++ {
					work.Set((*m)+i-1, d.Get(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Dgemv(NoTrans, m, n, toPtrf64(-one), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), &one, work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dasum(m, work, func() *int { y := 1; return &y }()))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				goblas.Dcopy(m, a.Vector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*m); i++ {
					work.Set((*m)+i-1, d.Get(i-1)*pt.Get(i-1, j-1))
				}
				goblas.Dgemv(NoTrans, m, m, toPtrf64(-one), q, ldq, work.Off((*m)+1-1), func() *int { y := 1; return &y }(), &one, work, func() *int { y := 1; return &y }())
				(*resid) = maxf64(*resid, goblas.Dasum(m, work, func() *int { y := 1; return &y }()))
			}
		}
	}

	//     Compute norm(A - Q * B * P') / ( n * norm(A) * EPS )
	anorm = golapack.Dlange('1', m, n, a, lda, work)
	eps = golapack.Dlamch(Precision)
	//
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
