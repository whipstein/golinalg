package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt17 computes the ratio
//
//    || R'*op(A) ||/(||A||*alpha*max(M,N,NRHS)*eps)
//
// where R = op(A)*X - B, op(A) is A or A', and
//
//    alpha = ||B|| if IRESID = 1 (zero-residual problem)
//    alpha = ||R|| if IRESID = 2 (otherwise).
func Dqrt17(trans byte, iresid, m, n, nrhs *int, a *mat.Matrix, lda *int, x *mat.Matrix, ldx *int, b *mat.Matrix, ldb *int, c *mat.Matrix, work *mat.Vector, lwork *int) (dqrt17Return float64) {
	var err, norma, normb, normrs, one, smlnum, zero float64
	var info, iscl, ncols, nrows int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqrt17Return = zero

	if trans == 'N' {
		nrows = (*m)
		ncols = (*n)
	} else if trans == 'T' {
		nrows = (*n)
		ncols = (*m)
	} else {
		gltest.Xerbla([]byte("DQRT17"), 1)
		return
	}

	if (*lwork) < ncols*(*nrhs) {
		gltest.Xerbla([]byte("DQRT17"), 13)
		return
	}

	if (*m) <= 0 || (*n) <= 0 || (*nrhs) <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, lda, rwork)
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	// bignum = one / smlnum
	iscl = 0

	//     compute residual and scale it
	golapack.Dlacpy('A', &nrows, nrhs, b, ldb, c, ldb)
	goblas.Dgemm(mat.TransByte(trans), NoTrans, &nrows, nrhs, &ncols, toPtrf64(-one), a, lda, x, ldx, &one, c, ldb)
	normrs = golapack.Dlange('M', &nrows, nrhs, c, ldb, rwork)
	if normrs > smlnum {
		iscl = 1
		golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &normrs, &one, &nrows, nrhs, c, ldb, &info)
	}

	//     compute R'*A
	goblas.Dgemm(Trans, mat.TransByte(trans), nrhs, &ncols, &nrows, &one, c, ldb, a, lda, &zero, work.Matrix(*nrhs, opts), nrhs)

	//     compute and properly scale error
	err = golapack.Dlange('O', nrhs, &ncols, work.Matrix(*nrhs, opts), nrhs, rwork)
	if norma != zero {
		err = err / norma
	}

	if iscl == 1 {
		err = err * normrs
	}

	if (*iresid) == 1 {
		normb = golapack.Dlange('O', &nrows, nrhs, b, ldb, rwork)
		if normb != zero {
			err = err / normb
		}
	} else {
		if normrs != zero {
			err = err / normrs
		}
	}

	dqrt17Return = err / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n, *nrhs)))
	return
}
