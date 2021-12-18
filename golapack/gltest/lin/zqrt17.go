package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt17 computes the ratio
//
//    || R'*op(A) ||/(||A||*alpha*max(M,N,NRHS)*eps)
//
// where R = op(A)*X - B, op(A) is A or A', and
//
//    alpha = ||B|| if IRESID = 1 (zero-residual problem)
//    alpha = ||R|| if IRESID = 2 (otherwise).
func zqrt17(trans mat.MatTrans, iresid, m, n, nrhs int, a, x, b, c *mat.CMatrix, work *mat.CVector, lwork int) (zqrt17Return float64) {
	var err2, norma, normb, normrs, one, smlnum, zero float64
	var iscl, ncols, nrows int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqrt17Return = zero

	if trans == NoTrans {
		nrows = m
		ncols = n
	} else if trans == ConjTrans {
		nrows = n
		ncols = m
	} else {
		gltest.Xerbla("zqrt17", 1)
		return
	}

	if lwork < ncols*nrhs {
		gltest.Xerbla("zqrt17", 13)
		return
	}

	if m <= 0 || n <= 0 || nrhs <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, rwork)
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	// bignum = one / smlnum
	iscl = 0

	//     compute residual and scale it
	golapack.Zlacpy(Full, nrows, nrhs, b, c)
	if err = c.Gemm(trans, NoTrans, nrows, nrhs, ncols, complex(-one, 0), a, x, complex(one, 0)); err != nil {
		panic(err)
	}
	normrs = golapack.Zlange('M', nrows, nrhs, c, rwork)
	if normrs > smlnum {
		iscl = 1
		if err = golapack.Zlascl('G', 0, 0, normrs, one, nrows, nrhs, c); err != nil {
			panic(err)
		}
	}

	//     compute R'*A
	if err = work.CMatrix(nrhs, opts).Gemm(ConjTrans, trans, nrhs, ncols, nrows, complex(one, 0), c, a, complex(zero, 0)); err != nil {
		panic(err)
	}

	//     compute and properly scale error
	err2 = golapack.Zlange('O', nrhs, ncols, work.CMatrix(nrhs, opts), rwork)
	if norma != zero {
		err2 = err2 / norma
	}

	if iscl == 1 {
		err2 = err2 * normrs
	}

	if iresid == 1 {
		normb = golapack.Zlange('O', nrows, nrhs, b, rwork)
		if normb != zero {
			err2 = err2 / normb
		}
	} else {
		if normrs != zero {
			err2 = err2 / normrs
		}
	}

	zqrt17Return = err2 / (golapack.Dlamch(Epsilon) * float64(max(m, n, nrhs)))
	return
}
