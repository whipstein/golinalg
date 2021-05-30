package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zqrt14 checks whether X is in the row space of A or A'.  It does so
// by scaling both X and A such that their norms are in the range
// [sqrt(eps), 1/sqrt(eps)], then computing a QR factorization of [A,X]
// (if TRANS = 'C') or an LQ factorization of [A',X]' (if TRANS = 'N'),
// and returning the norm of the trailing triangle, scaled by
// maxint(M,N,NRHS)*eps.
func Zqrt14(trans byte, m, n, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, work *mat.CVector, lwork *int) (zqrt14Return float64) {
	var tpsd bool
	var anrm, err, one, xnrm, zero float64
	var i, info, j, ldwork int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqrt14Return = zero
	if trans == 'N' {
		ldwork = (*m) + (*nrhs)
		tpsd = false
		if (*lwork) < ((*m)+(*nrhs))*((*n)+2) {
			gltest.Xerbla([]byte("ZQRT14"), 10)
			return
		} else if (*n) <= 0 || (*nrhs) <= 0 {
			return
		}
	} else if trans == 'C' {
		ldwork = (*m)
		tpsd = true
		if (*lwork) < ((*n)+(*nrhs))*((*m)+2) {
			gltest.Xerbla([]byte("ZQRT14"), 10)
			return
		} else if (*m) <= 0 || (*nrhs) <= 0 {
			return
		}
	} else {
		gltest.Xerbla([]byte("ZQRT14"), 1)
		return
	}

	//     Copy and scale A
	golapack.Zlacpy('A', m, n, a, lda, work.CMatrix(ldwork, opts), &ldwork)
	anrm = golapack.Zlange('M', m, n, work.CMatrix(ldwork, opts), &ldwork, rwork)
	if anrm != zero {
		golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &one, m, n, work.CMatrix(ldwork, opts), &ldwork, &info)
	}

	//     Copy X or X' into the right place and scale it
	if tpsd {
		//        Copy X into columns n+1:n+nrhs of work
		golapack.Zlacpy('A', m, nrhs, x, ldx, work.CMatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork)
		xnrm = golapack.Zlange('M', m, nrhs, work.CMatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork, rwork)
		if xnrm != zero {
			golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &xnrm, &one, m, nrhs, work.CMatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork, &info)
		}
		anrm = golapack.Zlange('O', m, toPtr((*n)+(*nrhs)), work.CMatrix(ldwork, opts), &ldwork, rwork)

		//        Compute QR factorization of X
		golapack.Zgeqr2(m, toPtr((*n)+(*nrhs)), work.CMatrix(ldwork, opts), &ldwork, work.Off(ldwork*((*n)+(*nrhs))+1-1), work.Off(ldwork*((*n)+(*nrhs))+minint(*m, (*n)+(*nrhs))+1-1), &info)

		//        Compute largest entry in upper triangle of
		//        work(n+1:m,n+1:n+nrhs)
		err = zero
		for j = (*n) + 1; j <= (*n)+(*nrhs); j++ {
			for i = (*n) + 1; i <= minint(*m, j); i++ {
				err = maxf64(err, work.GetMag(i+(j-1)*(*m)-1))
			}
		}

	} else {
		//        Copy X' into rows m+1:m+nrhs of work
		for i = 1; i <= (*n); i++ {
			for j = 1; j <= (*nrhs); j++ {
				work.Set((*m)+j+(i-1)*ldwork-1, x.GetConj(i-1, j-1))
			}
		}

		xnrm = golapack.Zlange('M', nrhs, n, work.CMatrixOff((*m)+1-1, ldwork, opts), &ldwork, rwork)
		if xnrm != zero {
			golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &xnrm, &one, nrhs, n, work.CMatrixOff((*m)+1-1, ldwork, opts), &ldwork, &info)
		}

		//        Compute LQ factorization of work
		golapack.Zgelq2(&ldwork, n, work.CMatrix(ldwork, opts), &ldwork, work.Off(ldwork*(*n)+1-1), work.Off(ldwork*((*n)+1)+1-1), &info)

		//        Compute largest entry in lower triangle in
		//        work(m+1:m+nrhs,m+1:n)
		err = zero
		for j = (*m) + 1; j <= (*n); j++ {
			for i = j; i <= ldwork; i++ {
				err = maxf64(err, work.GetMag(i+(j-1)*ldwork-1))
			}
		}

	}

	zqrt14Return = err / (float64(maxint(*m, *n, *nrhs)) * golapack.Dlamch(Epsilon))

	return
}
