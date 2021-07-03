package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt17 computes the ratio
//
//    || R'*op(A) ||/(||A||*alpha*maxint(M,N,NRHS)*eps)
//
// where R = op(A)*X - B, op(A) is A or A', and
//
//    alpha = ||B|| if IRESID = 1 (zero-residual problem)
//    alpha = ||R|| if IRESID = 2 (otherwise).
func Zqrt17(trans byte, iresid, m, n, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, c *mat.CMatrix, work *mat.CVector, lwork *int) (zqrt17Return float64) {
	var err2, norma, normb, normrs, one, smlnum, zero float64
	var info, iscl, ncols, nrows int
	var err error
	_ = err

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqrt17Return = zero

	if trans == 'N' {
		nrows = (*m)
		ncols = (*n)
	} else if trans == 'C' {
		nrows = (*n)
		ncols = (*m)
	} else {
		gltest.Xerbla([]byte("ZQRT17"), 1)
		return
	}

	if (*lwork) < ncols*(*nrhs) {
		gltest.Xerbla([]byte("ZQRT17"), 13)
		return
	}

	if (*m) <= 0 || (*n) <= 0 || (*nrhs) <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, lda, rwork)
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	// bignum = one / smlnum
	iscl = 0

	//     compute residual and scale it
	golapack.Zlacpy('A', &nrows, nrhs, b, ldb, c, ldb)
	err = goblas.Zgemm(mat.TransByte(trans), NoTrans, nrows, *nrhs, ncols, complex(-one, 0), a, *lda, x, *ldx, complex(one, 0), c, *ldb)
	normrs = golapack.Zlange('M', &nrows, nrhs, c, ldb, rwork)
	if normrs > smlnum {
		iscl = 1
		golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &normrs, &one, &nrows, nrhs, c, ldb, &info)
	}

	//     compute R'*A
	err = goblas.Zgemm(ConjTrans, mat.TransByte(trans), *nrhs, ncols, nrows, complex(one, 0), c, *ldb, a, *lda, complex(zero, 0), work.CMatrix(*nrhs, opts), *nrhs)

	//     compute and properly scale error
	err2 = golapack.Zlange('O', nrhs, &ncols, work.CMatrix(*nrhs, opts), nrhs, rwork)
	if norma != zero {
		err2 = err2 / norma
	}

	if iscl == 1 {
		err2 = err2 * normrs
	}

	if (*iresid) == 1 {
		normb = golapack.Zlange('O', &nrows, nrhs, b, ldb, rwork)
		if normb != zero {
			err2 = err2 / normb
		}
	} else {
		if normrs != zero {
			err2 = err2 / normrs
		}
	}

	zqrt17Return = err2 / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n, *nrhs)))
	return
}
