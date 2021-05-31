package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt14 checks whether X is in the row space of A or A'.  It does so
// by scaling both X and A such that their norms are in the range
// [sqrt(eps), 1/sqrt(eps)], then computing a QR factorization of [A,X]
// (if TRANS = 'T') or an LQ factorization of [A',X]' (if TRANS = 'N'),
// and returning the norm of the trailing triangle, scaled by
// MAX(M,N,NRHS)*eps.
func Dqrt14(trans byte, m, n, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, work *mat.Vector, lwork *int) (dqrt14Return float64) {
	var tpsd bool
	var anrm, err, one, xnrm, zero float64
	var i, info, j, ldwork int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqrt14Return = zero
	if trans == 'N' {
		ldwork = (*m) + (*nrhs)
		tpsd = false
		if (*lwork) < ((*m)+(*nrhs))*((*n)+2) {
			gltest.Xerbla([]byte("DQRT14"), 10)
			return
		} else if (*n) <= 0 || (*nrhs) <= 0 {
			return
		}
	} else if trans == 'T' {
		ldwork = (*m)
		tpsd = true
		if (*lwork) < ((*n)+(*nrhs))*((*m)+2) {
			gltest.Xerbla([]byte("DQRT14"), 10)
			return
		} else if (*m) <= 0 || (*nrhs) <= 0 {
			return
		}
	} else {
		gltest.Xerbla([]byte("DQRT14"), 1)
		return
	}

	//     Copy and scale A
	golapack.Dlacpy('A', m, n, a, lda, work.Matrix(ldwork, opts), &ldwork)
	anrm = golapack.Dlange('M', m, n, work.Matrix(ldwork, opts), &ldwork, rwork)
	if anrm != zero {
		golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &one, m, n, work.Matrix(ldwork, opts), &ldwork, &info)
	}

	//     Copy X or X' into the right place and scale it
	if tpsd {
		//        Copy X into columns n+1:n+nrhs of work
		golapack.Dlacpy('A', m, nrhs, x, ldx, work.MatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork)
		xnrm = golapack.Dlange('M', m, nrhs, work.MatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork, rwork)
		if xnrm != zero {
			golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &xnrm, &one, m, nrhs, work.MatrixOff((*n)*ldwork+1-1, ldwork, opts), &ldwork, &info)
		}
		anrm = golapack.Dlange('O', m, toPtr((*n)+(*nrhs)), work.Matrix(ldwork, opts), &ldwork, rwork)

		//        Compute QR factorization of X
		golapack.Dgeqr2(m, toPtr((*n)+(*nrhs)), work.Matrix(ldwork, opts), &ldwork, work.Off(ldwork*((*n)+(*nrhs))+1-1), work.Off(ldwork*((*n)+(*nrhs))+minint(*m, (*n)+(*nrhs))+1-1), &info)

		//        Compute largest entry in upper triangle of
		//        work(n+1:m,n+1:n+nrhs)
		err = zero
		for j = (*n) + 1; j <= (*n)+(*nrhs); j++ {
			for i = (*n) + 1; i <= minint(*m, j); i++ {
				err = maxf64(err, math.Abs(work.Get(i+(j-1)*(*m)-1)))
			}
		}

	} else {
		//        Copy X' into rows m+1:m+nrhs of work
		for i = 1; i <= (*n); i++ {
			for j = 1; j <= (*nrhs); j++ {
				work.Set((*m)+j+(i-1)*ldwork-1, x.Get(i-1, j-1))
			}
		}

		xnrm = golapack.Dlange('M', nrhs, n, work.MatrixOff((*m)+1-1, ldwork, opts), &ldwork, rwork)
		if xnrm != zero {
			golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &xnrm, &one, nrhs, n, work.MatrixOff((*m)+1-1, ldwork, opts), &ldwork, &info)
		}

		//        Compute LQ factorization of work
		golapack.Dgelq2(&ldwork, n, work.Matrix(ldwork, opts), &ldwork, work.Off(ldwork*(*n)+1-1), work.Off(ldwork*((*n)+1)+1-1), &info)

		//        Compute largest entry in lower triangle in
		//        work(m+1:m+nrhs,m+1:n)
		err = zero
		for j = (*m) + 1; j <= (*n); j++ {
			for i = j; i <= ldwork; i++ {
				err = maxf64(err, math.Abs(work.Get(i+(j-1)*ldwork-1)))
			}
		}

	}

	dqrt14Return = err / (float64(maxint(*m, *n, *nrhs)) * golapack.Dlamch(Epsilon))

	return
}
