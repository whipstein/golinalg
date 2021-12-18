package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt14 checks whether X is in the row space of A or A'.  It does so
// by scaling both X and A such that their norms are in the range
// [sqrt(eps), 1/sqrt(eps)], then computing a QR factorization of [A,X]
// (if TRANS = 'T') or an LQ factorization of [A',X]' (if TRANS = 'N'),
// and returning the norm of the trailing triangle, scaled by
// MAX(M,N,NRHS)*eps.
func dqrt14(trans mat.MatTrans, m, n, nrhs int, a, x *mat.Matrix, work *mat.Vector, lwork int) (dqrt14Return float64) {
	var tpsd bool
	var anrm, err, one, xnrm, zero float64
	var i, j, ldwork int
	var err2 error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqrt14Return = zero
	if trans == NoTrans {
		ldwork = m + nrhs
		tpsd = false
		if lwork < (m+nrhs)*(n+2) {
			gltest.Xerbla("dqrt14", 10)
			return
		} else if n <= 0 || nrhs <= 0 {
			return
		}
	} else if trans == Trans {
		ldwork = m
		tpsd = true
		if lwork < (n+nrhs)*(m+2) {
			gltest.Xerbla("dqrt14", 10)
			return
		} else if m <= 0 || nrhs <= 0 {
			return
		}
	} else {
		gltest.Xerbla("dqrt14", 1)
		return
	}

	//     Copy and scale A
	golapack.Dlacpy(Full, m, n, a, work.Matrix(ldwork, opts))
	anrm = golapack.Dlange('M', m, n, work.Matrix(ldwork, opts), rwork)
	if anrm != zero {
		if err2 = golapack.Dlascl('G', 0, 0, anrm, one, m, n, work.Matrix(ldwork, opts)); err2 != nil {
			panic(err2)
		}
	}

	//     Copy X or X' into the right place and scale it
	if tpsd {
		//        Copy X into columns n+1:n+nrhs of work
		golapack.Dlacpy(Full, m, nrhs, x, work.Off(n*ldwork).Matrix(ldwork, opts))
		xnrm = golapack.Dlange('M', m, nrhs, work.Off(n*ldwork).Matrix(ldwork, opts), rwork)
		if xnrm != zero {
			if err2 = golapack.Dlascl('G', 0, 0, xnrm, one, m, nrhs, work.Off(n*ldwork).Matrix(ldwork, opts)); err2 != nil {
				panic(err2)
			}
		}
		anrm = golapack.Dlange('O', m, n+nrhs, work.Matrix(ldwork, opts), rwork)

		//        Compute QR factorization of X
		if err2 = golapack.Dgeqr2(m, n+nrhs, work.Matrix(ldwork, opts), work.Off(ldwork*(n+nrhs)), work.Off(ldwork*(n+nrhs)+min(m, n+nrhs))); err2 != nil {
			panic(err2)
		}

		//        Compute largest entry in upper triangle of
		//        work(n+1:m,n+1:n+nrhs)
		err = zero
		for j = n + 1; j <= n+nrhs; j++ {
			for i = n + 1; i <= min(m, j); i++ {
				err = math.Max(err, math.Abs(work.Get(i+(j-1)*m-1)))
			}
		}

	} else {
		//        Copy X' into rows m+1:m+nrhs of work
		for i = 1; i <= n; i++ {
			for j = 1; j <= nrhs; j++ {
				work.Set(m+j+(i-1)*ldwork-1, x.Get(i-1, j-1))
			}
		}

		xnrm = golapack.Dlange('M', nrhs, n, work.Off(m).Matrix(ldwork, opts), rwork)
		if xnrm != zero {
			if err2 = golapack.Dlascl('G', 0, 0, xnrm, one, nrhs, n, work.Off(m).Matrix(ldwork, opts)); err2 != nil {
				panic(err2)
			}
		}

		//        Compute LQ factorization of work
		if err2 = golapack.Dgelq2(ldwork, n, work.Matrix(ldwork, opts), work.Off(ldwork*n), work.Off(ldwork*(n+1))); err2 != nil {
			panic(err)
		}

		//        Compute largest entry in lower triangle in
		//        work(m+1:m+nrhs,m+1:n)
		err = zero
		for j = m + 1; j <= n; j++ {
			for i = j; i <= ldwork; i++ {
				err = math.Max(err, math.Abs(work.Get(i+(j-1)*ldwork-1)))
			}
		}

	}

	dqrt14Return = err / (float64(max(m, n, nrhs)) * golapack.Dlamch(Epsilon))

	return
}
