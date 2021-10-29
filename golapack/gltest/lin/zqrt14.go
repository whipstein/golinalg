package lin

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt14 checks whether X is in the row space of A or A'.  It does so
// by scaling both X and A such that their norms are in the range
// [sqrt(eps), 1/sqrt(eps)], then computing a QR factorization of [A,X]
// (if TRANS = 'C') or an LQ factorization of [A',X]' (if TRANS = 'N'),
// and returning the norm of the trailing triangle, scaled by
// max(M,N,NRHS)*eps.
func zqrt14(trans mat.MatTrans, m, n, nrhs int, a, x *mat.CMatrix, work *mat.CVector, lwork int) (zqrt14Return float64) {
	var tpsd bool
	var anrm, errf, one, xnrm, zero float64
	var i, j, ldwork int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqrt14Return = zero
	if trans == NoTrans {
		ldwork = m + nrhs
		tpsd = false
		if lwork < (m+nrhs)*(n+2) {
			err = fmt.Errorf("lwork < (m+nrhs)*(n+2): lwork=%v, m=%v, n=%v, nrhs=%v", lwork, m, n, nrhs)
			gltest.Xerbla2("zqrt14", err)
			return
		} else if n <= 0 || nrhs <= 0 {
			return
		}
	} else if trans == ConjTrans {
		ldwork = m
		tpsd = true
		if lwork < (n+nrhs)*(m+2) {
			err = fmt.Errorf("lwork < (n+nrhs)*(m+2): lwork=%v, m=%v, n=%v, nrhs=%v", lwork, m, n, nrhs)
			gltest.Xerbla2("zqrt14", err)
			return
		} else if m <= 0 || nrhs <= 0 {
			return
		}
	} else {
		gltest.Xerbla("zqrt14", 1)
		return
	}

	//     Copy and scale A
	golapack.Zlacpy(Full, m, n, a, work.CMatrix(ldwork, opts))
	anrm = golapack.Zlange('M', m, n, work.CMatrix(ldwork, opts), rwork)
	if anrm != zero {
		if err = golapack.Zlascl('G', 0, 0, anrm, one, m, n, work.CMatrix(ldwork, opts)); err != nil {
			panic(err)
		}
	}

	//     Copy X or X' into the right place and scale it
	if tpsd {
		//        Copy X into columns n+1:n+nrhs of work
		golapack.Zlacpy(Full, m, nrhs, x, work.CMatrixOff(n*ldwork, ldwork, opts))
		xnrm = golapack.Zlange('M', m, nrhs, work.CMatrixOff(n*ldwork, ldwork, opts), rwork)
		if xnrm != zero {
			if err = golapack.Zlascl('G', 0, 0, xnrm, one, m, nrhs, work.CMatrixOff(n*ldwork, ldwork, opts)); err != nil {
				panic(err)
			}
		}
		anrm = golapack.Zlange('O', m, n+nrhs, work.CMatrix(ldwork, opts), rwork)

		//        Compute QR factorization of X
		if err = golapack.Zgeqr2(m, n+nrhs, work.CMatrix(ldwork, opts), work.Off(ldwork*(n+nrhs)), work.Off(ldwork*(n+nrhs)+min(m, n+nrhs))); err != nil {
			panic(err)
		}

		//        Compute largest entry in upper triangle of
		//        work(n+1:m,n+1:n+nrhs)
		errf = zero
		for j = n + 1; j <= n+nrhs; j++ {
			for i = n + 1; i <= min(m, j); i++ {
				errf = math.Max(errf, work.GetMag(i+(j-1)*m-1))
			}
		}

	} else {
		//        Copy X' into rows m+1:m+nrhs of work
		for i = 1; i <= n; i++ {
			for j = 1; j <= nrhs; j++ {
				work.Set(m+j+(i-1)*ldwork-1, x.GetConj(i-1, j-1))
			}
		}

		xnrm = golapack.Zlange('M', nrhs, n, work.CMatrixOff(m, ldwork, opts), rwork)
		if xnrm != zero {
			if err = golapack.Zlascl('G', 0, 0, xnrm, one, nrhs, n, work.CMatrixOff(m, ldwork, opts)); err != nil {
				panic(err)
			}
		}

		//        Compute LQ factorization of work
		if err = golapack.Zgelq2(ldwork, n, work.CMatrix(ldwork, opts), work.Off(ldwork*n), work.Off(ldwork*(n+1))); err != nil {
			panic(err)
		}

		//        Compute largest entry in lower triangle in
		//        work(m+1:m+nrhs,m+1:n)
		errf = zero
		for j = m + 1; j <= n; j++ {
			for i = j; i <= ldwork; i++ {
				errf = math.Max(errf, work.GetMag(i+(j-1)*ldwork-1))
			}
		}

	}

	zqrt14Return = errf / (float64(max(m, n, nrhs)) * golapack.Dlamch(Epsilon))

	return
}
